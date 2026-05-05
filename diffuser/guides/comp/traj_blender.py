import numpy as np
import diffuser.utils as utils
from diffuser.models.cd_stgl_sml_dfu import Stgl_Sml_GauDiffusion_InvDyn_V1
from diffuser.datasets.normalization import DatasetNormalizer
from diffuser.guides.comp.cd_sml_policies import StitchSearchNode, StitchSearchResult

class Traj_Blender:
    def __init__(self, diffusion: Stgl_Sml_GauDiffusion_InvDyn_V1, 
                        normalizer: DatasetNormalizer,
                        blend_type: str,
                        exp_beta=3,
                        search_beam_width=6,
                        search_chunk_pool=4,
                        search_density_weight=0.35,
                        search_overlap_weight=1.0,
                        search_vel_weight=0.35,
                        search_acc_weight=0.15,
                        search_rough_weight=0.05,
                        search_commit_weight=0.0,
                        search_center_ratio=0.5,
                        search_edit_weight=0.15,
                        search_density_gate_temp=0.25,
                        search_hold_ratio=0.25,
                        ):
        self.exp_beta = exp_beta
        self.diffusion = diffusion
        self.blend_type = blend_type
        self.len_ovlp = self.diffusion.len_ovlp_cd
        self.hzn_step_size = self.diffusion.horizon - self.len_ovlp
        self.hzn = self.diffusion.horizon
        self.gap_len = self.hzn - 2 * self.len_ovlp
        self.normalizer = normalizer
        self.search_beam_width = search_beam_width
        self.search_chunk_pool = search_chunk_pool
        self.search_density_weight = search_density_weight
        self.search_overlap_weight = search_overlap_weight
        self.search_vel_weight = search_vel_weight
        self.search_acc_weight = search_acc_weight
        self.search_rough_weight = search_rough_weight
        self.search_commit_weight = search_commit_weight
        self.search_center_ratio = search_center_ratio
        self.search_edit_weight = search_edit_weight
        self.search_density_gate_temp = search_density_gate_temp
        self.search_hold_ratio = search_hold_ratio
        self.search_ops = (
            "exp",
            "cosine",
            "smoothstep",
            "min_jerk",
            "hold_bridge",
            "mode_patch",
            "density_gate",
        )
        assert self.gap_len > 0

    def blend_traj_lists(self, trajs_list, do_unnorm):
        """
        trajs_list: list of len n_comp: [ (B,H,D),..., ]
        returns:
            - trajs_out: a list of [tot_hzn, dim]
        """
        if self.blend_type in ["tree", "tree_search", "meta_tree"]:
            raise ValueError("tree search blending requires search_blend_traj_lists")

        ## to np and unnorm
        trajs_list = utils.get_np_trajs_list(trajs_list, do_unnorm=do_unnorm, 
                                             normalizer=self.normalizer)

        n_comp = len(trajs_list)
        b_s,_, dd = trajs_list[0].shape ## b h d

        tot_hzn = n_comp * self.diffusion.horizon - \
                    (n_comp - 1) * self.diffusion.len_ovlp_cd
        
        print(f'{tot_hzn=}')
        # trajs_out = np.zeros( shape=(b_s, tot_hzn, dd) ) ## NOTE: default is float64
        trajs_out = np.zeros( shape=(b_s, tot_hzn, dd), dtype=np.float32 ) ## Dec 2: changed to float32
        cnt_v = np.zeros_like(trajs_out)
        ## copy non-ovlp parts
        for i_c in range(n_comp):
            tjs_p_i = trajs_list[i_c]

            if i_c == 0:
                tmp_idx_1 = 0
                tmp_idx_2 = self.hzn_step_size
                ## B,hstep,dim
                trajs_out[:, tmp_idx_1:tmp_idx_2, :] = tjs_p_i[:, :self.hzn_step_size, :]
            elif i_c < n_comp - 1:
                tmp_idx_1 = self.hzn + (i_c - 1) * self.hzn_step_size
                tmp_idx_2 = tmp_idx_1 + self.gap_len
                trajs_out[:, tmp_idx_1:tmp_idx_2, :] = tjs_p_i[:, self.len_ovlp:self.len_ovlp+self.gap_len, :]
                
            elif i_c == n_comp - 1:
                tmp_idx_1 = self.hzn + (i_c - 1) * self.hzn_step_size
                tmp_idx_2 = tmp_idx_1 + self.hzn_step_size

                assert tmp_idx_2 == tot_hzn
                trajs_out[:, tmp_idx_1:tmp_idx_2, :] = tjs_p_i[:, self.len_ovlp:, :]

            cnt_v[ :, tmp_idx_1:tmp_idx_2, : ] += 1
            utils.print_color(f'{i_c=} {tmp_idx_1=}, {tmp_idx_2=}, {tot_hzn=}')

        ## handle and merge the ovlp parts
        for i_c in range(n_comp-1):
            tmp_idx_1 = (i_c + 1) * self.hzn_step_size
            tmp_idx_2 = tmp_idx_1 + self.len_ovlp

            ## b,sm_hzn,d
            tjs_p_i = trajs_list[i_c]
            _, end_tjs_i = self.diffusion.extract_ovlp_from_full(tjs_p_i)
            ## b,sm_hzn,d
            tjs_p_i_plus_1 = trajs_list[i_c+1]
            st_tjs_i_plus_1, _ = self.diffusion.extract_ovlp_from_full(tjs_p_i_plus_1)


            ## b,len_o,d
            trajs_blend = blend_2_np_trajs_23d(end_tjs_i, st_tjs_i_plus_1, 
                                               self.blend_type, self.exp_beta)

            trajs_out[:, tmp_idx_1:tmp_idx_2, :] = trajs_blend
            cnt_v[:, tmp_idx_1:tmp_idx_2, :] += 1

            utils.print_color(f'{i_c=} {tmp_idx_1=}, {tmp_idx_2=}')
        assert tmp_idx_2 == (tot_hzn - self.hzn_step_size)


        assert (cnt_v == 1).all()

        

        return trajs_out

    def search_blend_traj_lists(self, trajs_list_np_un, density_scores):
        """
        Tree search over chunk candidates and seam operators.

        Args:
            trajs_list_np_un: list of len n_comp, each element (B, H, D), unnormalized.
            density_scores: np.ndarray, shape (n_comp, B), lower is denser.
        """
        n_comp = len(trajs_list_np_un)
        assert n_comp >= 2
        assert density_scores.shape[0] == n_comp

        candidate_order = [
            np.argsort(density_scores[i_c])[: min(self.search_chunk_pool, trajs_list_np_un[i_c].shape[0])]
            for i_c in range(n_comp)
        ]

        beam = [
            StitchSearchNode(
                score=-self.search_density_weight * float(density_scores[0, idx]),
                chunk_indices=(int(idx),),
                operators=(),
                seam_scores=(),
            )
            for idx in candidate_order[0]
        ]
        beam = sorted(beam, key=lambda node: node.score, reverse=True)[
            : self.search_beam_width
        ]

        for i_c in range(1, n_comp):
            next_beam = []
            for node in beam:
                prev_idx = node.chunk_indices[-1]
                prev_chunk = trajs_list_np_un[i_c - 1][prev_idx]
                prev_density = float(density_scores[i_c - 1, prev_idx])
                for cur_idx in candidate_order[i_c]:
                    cur_idx = int(cur_idx)
                    cur_chunk = trajs_list_np_un[i_c][cur_idx]
                    cur_density = float(density_scores[i_c, cur_idx])
                    for op_name in self.search_ops:
                        _, seam_metrics = self.blend_pair(
                            prev_chunk,
                            cur_chunk,
                            blend_type=op_name,
                            density_pair=(prev_density, cur_density),
                            return_metrics=True,
                        )
                        seam_penalty = (
                            self.search_overlap_weight * seam_metrics["fit_mse"]
                            + self.search_vel_weight * seam_metrics["vel_mse"]
                            + self.search_acc_weight * seam_metrics["acc_mse"]
                            + self.search_rough_weight * seam_metrics["rough_mse"]
                        )
                        total_score = (
                            node.score
                            - self.search_density_weight * cur_density
                            - seam_penalty
                        )
                        next_beam.append(
                            StitchSearchNode(
                                score=total_score,
                                chunk_indices=node.chunk_indices + (cur_idx,),
                                operators=node.operators + (op_name,),
                                seam_scores=node.seam_scores + (-seam_penalty,),
                            )
                        )

            if len(next_beam) == 0:
                raise RuntimeError("tree search failed to expand any stitch candidates")
            beam = sorted(next_beam, key=lambda node: node.score, reverse=True)[
                : self.search_beam_width
            ]

        best = beam[0]
        selected_chunks = [
            trajs_list_np_un[i_c][best.chunk_indices[i_c]] for i_c in range(n_comp)
        ]
        selected_densities = [
            float(density_scores[i_c, best.chunk_indices[i_c]]) for i_c in range(n_comp)
        ]
        blended = self.compose_single_traj(
            selected_chunks,
            best.operators,
            selected_densities,
        )
        diagnostics = dict(
            chunk_indices=best.chunk_indices,
            operators=best.operators,
            seam_scores=best.seam_scores,
            chunk_density=tuple(selected_densities),
            candidate_order=[tuple(int(v) for v in order) for order in candidate_order],
        )
        return StitchSearchResult(
            blended_traj=blended,
            score=best.score,
            chunk_indices=best.chunk_indices,
            operators=best.operators,
            diagnostics=diagnostics,
        )

    def compose_single_traj(self, chunk_list, seam_ops, density_list=None):
        assert len(chunk_list) >= 2
        assert len(seam_ops) == len(chunk_list) - 1
        dd = chunk_list[0].shape[-1]
        tot_hzn = len(chunk_list) * self.hzn - (len(chunk_list) - 1) * self.len_ovlp
        traj_out = np.zeros((tot_hzn, dd), dtype=np.float32)

        for i_c, chunk in enumerate(chunk_list):
            if i_c == 0:
                tmp_idx_1 = 0
                tmp_idx_2 = self.hzn_step_size
                traj_out[tmp_idx_1:tmp_idx_2, :] = chunk[: self.hzn_step_size, :]
            elif i_c < len(chunk_list) - 1:
                tmp_idx_1 = self.hzn + (i_c - 1) * self.hzn_step_size
                tmp_idx_2 = tmp_idx_1 + self.gap_len
                traj_out[tmp_idx_1:tmp_idx_2, :] = chunk[
                    self.len_ovlp : self.len_ovlp + self.gap_len, :
                ]
            else:
                tmp_idx_1 = self.hzn + (i_c - 1) * self.hzn_step_size
                tmp_idx_2 = tmp_idx_1 + self.hzn_step_size
                traj_out[tmp_idx_1:tmp_idx_2, :] = chunk[self.len_ovlp :, :]

        for i_c in range(len(chunk_list) - 1):
            density_pair = None
            if density_list is not None:
                density_pair = (density_list[i_c], density_list[i_c + 1])
            overlap = self.blend_pair(
                chunk_list[i_c],
                chunk_list[i_c + 1],
                blend_type=seam_ops[i_c],
                density_pair=density_pair,
                return_metrics=False,
            )
            tmp_idx_1 = (i_c + 1) * self.hzn_step_size
            tmp_idx_2 = tmp_idx_1 + self.len_ovlp
            traj_out[tmp_idx_1:tmp_idx_2, :] = overlap

        return traj_out

    def blend_pair(self, traj_1, traj_2, blend_type=None, density_pair=None, return_metrics=False):
        blend_type = self.blend_type if blend_type is None else blend_type
        assert traj_1.ndim == 2 and traj_2.ndim == 2
        _, end_tjs_i = self.diffusion.extract_ovlp_from_full(traj_1[None, ...])
        st_tjs_i_plus_1, _ = self.diffusion.extract_ovlp_from_full(traj_2[None, ...])
        end_tjs_i = end_tjs_i[0]
        st_tjs_i_plus_1 = st_tjs_i_plus_1[0]

        if blend_type in ["min_jerk", "minjerk", "bridge"]:
            blend = self._blend_min_jerk(end_tjs_i, st_tjs_i_plus_1)
        elif blend_type in ["hold_bridge", "holdbridge", "hold"]:
            blend = self._blend_hold_bridge(
                end_tjs_i,
                st_tjs_i_plus_1,
                hold_ratio=self.search_hold_ratio,
            )
        elif blend_type in ["mode_patch", "modepatch", "patch"]:
            assert density_pair is not None, "mode_patch requires density_pair"
            blend = self._blend_mode_patch(
                end_tjs_i,
                st_tjs_i_plus_1,
                density_pair=density_pair,
                guard_ratio=self.search_hold_ratio,
            )
        else:
            blend = blend_2_np_trajs_23d(
                end_tjs_i,
                st_tjs_i_plus_1,
                blend_type=blend_type,
                beta=self.exp_beta,
                density_pair=density_pair,
                density_temp=self.search_density_gate_temp,
            )

        if not return_metrics:
            return blend.astype(np.float32)

        metrics = self._compute_seam_metrics(
            traj_1,
            traj_2,
            end_tjs_i,
            st_tjs_i_plus_1,
            blend,
        )
        return blend.astype(np.float32), metrics

    def _blend_min_jerk(self, end_traj, st_traj):
        len_tj = len(end_traj)
        if len_tj <= 2:
            return 0.5 * (end_traj + st_traj)

        p0 = end_traj[0]
        p1 = st_traj[-1]
        v0 = self._estimate_side_velocity(end_traj, side="start")
        v1 = self._estimate_side_velocity(st_traj, side="end")
        scale = max(len_tj - 1, 1)
        v0 = v0 * scale
        v1 = v1 * scale
        a0 = np.zeros_like(v0)
        a1 = np.zeros_like(v1)

        c0 = p0
        c1 = v0
        c2 = 0.5 * a0
        delta = p1 - (c0 + c1 + c2)
        c3 = 10 * delta - 4 * v1 + 0.5 * a1 - 6 * c1 - 3 * a0
        c4 = -15 * delta + 7 * v1 - a1 + 8 * c1 + 3 * a0
        c5 = 6 * delta - 3 * v1 + 0.5 * a1 - 3 * c1 - a0

        s = np.linspace(0.0, 1.0, len_tj, dtype=np.float32)[:, None]
        blend = (
            c0
            + c1 * s
            + c2 * (s**2)
            + c3 * (s**3)
            + c4 * (s**4)
            + c5 * (s**5)
        )
        return blend.astype(np.float32)

    def _blend_hold_bridge(self, end_traj, st_traj, hold_ratio=0.25):
        len_tj = len(end_traj)
        if len_tj <= 4:
            return self._blend_min_jerk(end_traj, st_traj)

        max_hold = max(1, (len_tj - 2) // 2)
        hold = int(round(len_tj * hold_ratio))
        hold = max(1, min(max_hold, hold))

        start_idx = hold - 1
        end_idx = len_tj - hold
        bridge_steps = end_idx - start_idx + 1
        if bridge_steps <= 2:
            return self._blend_min_jerk(end_traj, st_traj)

        blend = np.zeros_like(end_traj, dtype=np.float32)
        if start_idx > 0:
            blend[:start_idx] = end_traj[:start_idx]
        if end_idx + 1 < len_tj:
            blend[end_idx + 1 :] = st_traj[end_idx + 1 :]

        p0 = end_traj[start_idx]
        p1 = st_traj[end_idx]
        v0 = self._estimate_anchor_velocity(end_traj, start_idx)
        v1 = self._estimate_anchor_velocity(st_traj, end_idx)
        bridge = self._quintic_bridge(p0, p1, v0, v1, bridge_steps)
        blend[start_idx : end_idx + 1] = bridge
        return blend.astype(np.float32)

    def _blend_mode_patch(self, end_traj, st_traj, density_pair, guard_ratio=0.25):
        len_tj = len(end_traj)
        if len_tj <= 4:
            return blend_2_np_trajs_23d(
                end_traj,
                st_traj,
                blend_type="density_gate",
                beta=self.exp_beta,
                density_pair=density_pair,
                density_temp=self.search_density_gate_temp,
            )

        left_density, right_density = density_pair
        guard = int(round(len_tj * guard_ratio))
        guard = max(2, min(len_tj - 2, guard))
        ramp = max(2, min(len_tj // 4, guard))

        if left_density <= right_density:
            switch_center = len_tj - guard
        else:
            switch_center = guard

        switch_start = max(0, switch_center - ramp)
        switch_end = min(len_tj - 1, switch_center + ramp)
        if switch_end <= switch_start:
            switch_end = min(len_tj - 1, switch_start + 1)

        weights = np.ones((len_tj,), dtype=np.float32)
        if left_density <= right_density:
            weights[switch_end:] = 0.0
            span = max(1, switch_end - switch_start)
            x = np.linspace(0.0, 1.0, span + 1, dtype=np.float32)
            trans = 1.0 - (3.0 * x**2 - 2.0 * x**3)
            weights[switch_start : switch_end + 1] = trans
        else:
            weights[:switch_start] = 1.0
            weights[switch_end + 1 :] = 0.0
            span = max(1, switch_end - switch_start)
            x = np.linspace(0.0, 1.0, span + 1, dtype=np.float32)
            trans = 1.0 - (3.0 * x**2 - 2.0 * x**3)
            weights[switch_start : switch_end + 1] = trans

        blend = weights[:, None] * end_traj + (1.0 - weights[:, None]) * st_traj
        return blend.astype(np.float32)

    def _quintic_bridge(self, p0, p1, v0, v1, steps):
        scale = max(steps - 1, 1)
        v0 = v0 * scale
        v1 = v1 * scale
        a0 = np.zeros_like(v0)
        a1 = np.zeros_like(v1)

        c0 = p0
        c1 = v0
        c2 = 0.5 * a0
        delta = p1 - (c0 + c1 + c2)
        c3 = 10 * delta - 4 * v1 + 0.5 * a1 - 6 * c1 - 3 * a0
        c4 = -15 * delta + 7 * v1 - a1 + 8 * c1 + 3 * a0
        c5 = 6 * delta - 3 * v1 + 0.5 * a1 - 3 * c1 - a0

        s = np.linspace(0.0, 1.0, steps, dtype=np.float32)[:, None]
        bridge = (
            c0
            + c1 * s
            + c2 * (s**2)
            + c3 * (s**3)
            + c4 * (s**4)
            + c5 * (s**5)
        )
        return bridge.astype(np.float32)

    def _estimate_anchor_velocity(self, traj, idx):
        if idx <= 0:
            return traj[1] - traj[0]
        if idx >= len(traj) - 1:
            return traj[-1] - traj[-2]
        return 0.5 * (traj[idx + 1] - traj[idx - 1])

    def _estimate_side_velocity(self, traj, side):
        span = min(4, len(traj) - 1)
        if side == "start":
            diffs = np.diff(traj[: span + 1], axis=0)
        elif side == "end":
            diffs = np.diff(traj[-(span + 1) :], axis=0)
        else:
            raise ValueError(side)
        return diffs.mean(axis=0)

    def _compute_seam_metrics(self, traj_1, traj_2, end_traj, st_traj, blend):
        fit_mse = 0.5 * (
            np.mean((blend - end_traj) ** 2) + np.mean((blend - st_traj) ** 2)
        )

        center_len = int(round(len(blend) * self.search_center_ratio))
        center_len = max(2, min(len(blend), center_len))
        center_start = max(0, (len(blend) - center_len) // 2)
        center_end = center_start + center_len
        blend_center = blend[center_start:center_end]
        end_center = end_traj[center_start:center_end]
        st_center = st_traj[center_start:center_end]
        center_commit_mse = min(
            np.mean((blend_center - end_center) ** 2),
            np.mean((blend_center - st_center) ** 2),
        )
        center_gap_mse = np.mean((end_center - st_center) ** 2)

        left_ctx = traj_1[-self.len_ovlp - 1]
        right_ctx = traj_2[self.len_ovlp]
        vel_in = blend[0] - left_ctx
        vel_start = blend[1] - blend[0]
        vel_end = blend[-1] - blend[-2]
        vel_out = right_ctx - blend[-1]
        vel_mse = np.mean((vel_in - vel_start) ** 2) + np.mean(
            (vel_end - vel_out) ** 2
        )

        acc_start = left_ctx - 2.0 * blend[0] + blend[1]
        acc_end = blend[-2] - 2.0 * blend[-1] + right_ctx
        acc_mse = np.mean(acc_start**2) + np.mean(acc_end**2)

        if len(blend) > 2:
            rough = np.diff(blend, n=2, axis=0)
            rough_mse = np.mean(rough**2)
        else:
            rough_mse = 0.0

        return dict(
            fit_mse=float(fit_mse),
            center_commit_mse=float(center_commit_mse),
            center_gap_mse=float(center_gap_mse),
            vel_mse=float(vel_mse),
            acc_mse=float(acc_mse),
            rough_mse=float(rough_mse),
        )

def blend_2_np_trajs_23d(
    traj_1: np.ndarray,
    traj_2: np.ndarray,
    blend_type="exponential",
    beta=5,
    density_pair=None,
    density_temp=0.25,
):
    """
    ** Only takes in the ovlp parts **, blend full traj_1 and traj_2
    ** Blend for multiple dim,
    ----[----
         ----]-----

    Parameters:
    - traj_1: np.ndarray, shape (N1, D), first trajectory positions
    - traj_2: np.ndarray, shape (N2, D), second trajectory positions
    - blend_type: str, type of blending function ('exponential', 'cosine', 'linear', 'smoothstep')
    - beta: float, parameter for the exponential blending function (controls sharpness)

    Returns:
    - traj_blend: np.ndarray, blended trajectory positions
    """

    # assert traj_1.ndim == 2 and traj_1.shape[1] == 1
    assert traj_1.ndim in [2,3] and traj_2.ndim in [2,3] 
    assert traj_1.shape and traj_2.shape

    if traj_1.ndim == 2:
        len_tj, _ = traj_1.shape
    else:
        b_s, len_tj, _ = traj_1.shape
    # Overlapping region from t = 8 to t = 10
    t_overlap_start = 0
    t_overlap_end = len_tj - 1
    t_overlap = np.arange(0, len_tj) ## 1D

    ## Blending function selection, Checked, correct formula
    if blend_type in ['exponential', 'exp']:
        # Exponential blending function
        def w(t):
            exponent = -beta * (t - t_overlap_start) / (t_overlap_end - t_overlap_start)
            return (np.exp(exponent) - np.exp(-beta)) / (1 - np.exp(-beta))
    elif blend_type == 'cosine':
        # Cosine blending function
        def w(t):
            return 0.5 * (1 + np.cos(np.pi * (t - t_overlap_start) / (t_overlap_end - t_overlap_start)))
    elif blend_type == 'linear':
        # Linear blending function
        def w(t):
            return 1 - (t - t_overlap_start) / (t_overlap_end - t_overlap_start)
    elif blend_type == 'smoothstep':
        # Smoothstep blending function
        def w(t):
            x = (t - t_overlap_start) / (t_overlap_end - t_overlap_start)
            return 1 - (3 * x**2 - 2 * x**3)
    elif blend_type == "density_gate":
        assert density_pair is not None, "density_gate requires density_pair"
        left_density, right_density = density_pair
        left_score = 1.0 / max(left_density, 1e-6)
        right_score = 1.0 / max(right_density, 1e-6)
        shift = np.tanh((left_score - right_score) * density_temp)

        def w(t):
            x = (t - t_overlap_start) / (t_overlap_end - t_overlap_start)
            base = 1 - x
            return np.clip(base + 0.25 * shift, 0.0, 1.0)
    else:
        raise ValueError(
            "Invalid blending function. Choose 'exponential', 'cosine', "
            "'linear', 'smoothstep', or 'density_gate'."
        )

    ## Compute weights, np 1d, from 0 to 1
    ## weights = w(t_overlap)[:, np.newaxis]  # Column vector for broadcasting
    weights = w(t_overlap)  # Column vector for broadcasting
    if traj_1.ndim == 2:
        weights = weights[:, None] ## (len_tj, 1)
    elif traj_1.ndim == 3:
        weights = weights[None, :, None,] ## (1, len_tj, 1)
    
    # print(f'{weights[(0,-1),]=},{weights.shape=}')

    ## Blend the overlapping region
    traj_blend = weights * traj_1 + (1 - weights) * traj_2

    # print(f'{traj_blend.shape=}')

    return traj_blend
