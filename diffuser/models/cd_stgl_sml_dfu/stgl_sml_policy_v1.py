import numpy as np
import torch, einops, pdb, time
import diffuser.utils as utils
from diffuser.guides.comp.cd_sml_policies import Trajectories_invdyn

from diffuser.models.cd_stgl_sml_dfu import Stgl_Sml_GauDiffusion_InvDyn_V1
from diffuser.models.helpers import apply_conditioning
from diffuser.guides.comp.traj_blender import Traj_Blender
from diffuser.utils.cp_utils.plan_utils import split_trajs_list_by_prob
        

class Stgl_Sml_Policy_V1:

    def __init__(self, diffusion_model, 
                 normalizer, 
                 pol_config,
                 tj_blder_config,
                 ):
        """
        pick_type: how to pick from top_n
        """
        self.diffusion_model: Stgl_Sml_GauDiffusion_InvDyn_V1 = diffusion_model
        self.diffusion_model.eval() ## NOTE: must be the ema one
        self.normalizer = normalizer
        self.action_dim = normalizer.action_dim
        
        self.n_comp = pol_config['ev_n_comp']
        self.top_n = pol_config['ev_top_n'] ## 5
        self.pick_type = pol_config['ev_pick_type'] ## thresholding is also fine?
        assert self.pick_type in ['first', 'rand'] ## smallest dist or randomly pick one
        
        self.cp_infer_t_type = pol_config.get('ev_cp_infer_t_type', 'interleave')
        self.meta_method = pol_config.get('ev_meta_method', 'baseline')
        self.density_p_ratio = pol_config.get('ev_density_p_ratio', 0.35)
        self.density_n_mc = pol_config.get('ev_density_n_mc', 2)
        self.global_overlap_weight = pol_config.get('ev_global_overlap_weight', 1.0)
        self.risk_threshold = pol_config.get('ev_risk_threshold', 3.0)
        self.switch_margin = pol_config.get('ev_switch_margin', 0.5)
        self.repair_top_k = pol_config.get('ev_repair_top_k', 2)
        self.local_density_weight = pol_config.get('ev_local_density_weight', 0.0)
        self.local_focus_ratio = pol_config.get('ev_local_focus_ratio', 0.5)
        self.local_cost_guard = pol_config.get('ev_local_cost_guard', 0.0)
        self.local_rank_weight = pol_config.get('ev_local_rank_weight', 0.0)
        self.global_density_weight = pol_config.get('ev_global_density_weight', 0.0)
        self.global_density_n_mc = pol_config.get('ev_global_density_n_mc', 1)
        self.global_density_inter_rate = pol_config.get('ev_global_density_inter_rate', 1)
        self.global_density_t_mid = pol_config.get('ev_global_density_t_mid', 0.2)
        self.global_density_candidate_topk = pol_config.get('ev_global_density_candidate_topk', 8)
        self.global_density_proxy_type = pol_config.get(
            'ev_global_density_proxy_type', 'window_mean'
        )
        self.global_density_proxy_overlap_weight = pol_config.get(
            'ev_global_density_proxy_overlap_weight', 0.0
        )
        self.global_density_window_beta = pol_config.get(
            'ev_global_density_window_beta', 1.0
        )

        ## blender_type, exp_beta
        self.tj_blder = Traj_Blender(diffusion_model, normalizer, **tj_blder_config)
        if self.meta_method == 'cdgs':
            self.cp_infer_t_type = 'cdgs'
        elif self.global_density_weight > 0.0 or self.meta_method in ['global_density_guide', 'rcd']:
            self.cp_infer_t_type = 'same_t_p'
        self.diffusion_model.update_global_density_guide_config(
            dict(
                enabled=self.global_density_weight > 0.0,
                weight=float(self.global_density_weight),
                p_ratio=float(self.density_p_ratio),
                n_mc_samples=int(self.global_density_n_mc),
                inter_rate=int(self.global_density_inter_rate),
                t_mid=float(self.global_density_t_mid),
                use_normed_grad=True,
                exp_beta=float(self.tj_blder.exp_beta),
                proxy_type=str(self.global_density_proxy_type),
                overlap_weight=float(self.global_density_proxy_overlap_weight),
                window_beta=float(self.global_density_window_beta),
            )
        )
        self.ncp_pred_time_list = [] ## a list of tuple [n_comp, sampling_time]
        self.last_stitch_info = {}

    @property
    def device(self):
        parameters = list(self.diffusion_model.parameters())
        return parameters[0].device
    



    def gen_cond_stgl_parallel(self, g_cond, debug=False, b_s=1):
        """
        st_gl: *not normed*, np2d [2, ndim], e.g., [ [st], [end] ], [[2,1], [3,4]],
        b_s: batch_size, 10-20+
        """
        
        hzn = self.diffusion_model.horizon
        o_dim = self.diffusion_model.observation_dim ## TODO: obs_dim only?
        # c_shape = [b_s, hzn, o_dim] ## e.g.,(20,160,2)
        

        st_gl = g_cond['st_gl']
        st_gl = torch.tensor(self.normalizer.normalize(st_gl, 'observations'))
        
        
        ## shape: 2, n_probs, dim
        assert st_gl.ndim == 3 and st_gl.shape[0] == 2
        n_probs = st_gl.shape[1]
        
        c_shape = [b_s*n_probs, hzn, o_dim] ## e.g.,(b_s*n_p: 20*10=200,160,2)

        ## 0: tensor (n_parallel_probs,2); hzn-1: same
        ## make sure return is not a view
        stgl_cond = {
            0: einops.repeat(st_gl[0,:,:], 'n_p d -> (n_p rr) d', rr=b_s).clone(),
            hzn-1: einops.repeat(st_gl[1,:,:], 'n_p d -> (n_p rr) d', rr=b_s).clone(),
        }
        # pdb.set_trace() ## check if repeat is correct




        ## Run GPU Planning, x_dfu_all
        ## a list of len n_comp, elem: cuda tensor (b_s*n_p=200 or 400,sm_hzn,dim)
        if self.cp_infer_t_type == 'interleave': ## original our
            trajs_list_lg_b = self.diffusion_model.comp_pred_p_loop_n(
                c_shape, stgl_cond, n_comp=self.n_comp, return_diffusion=False)
        
        elif self.cp_infer_t_type == 'gsc': ## Jan 23
            trajs_list_lg_b = self.diffusion_model.comp_pred_p_loop_n_GSC(
                c_shape, stgl_cond, n_comp=self.n_comp, return_diffusion=False)
        elif self.cp_infer_t_type == 'cdgs':
            trajs_list_lg_b = self.diffusion_model.comp_pred_p_loop_n_CDGS(
                c_shape, stgl_cond, n_comp=self.n_comp, return_diffusion=False)
        else:
            raise NotImplementedError

        ## reshape to a list of len n_probs, each elem is a trajs_list

        ## a list of trajs_list(list) for each prob
        trajs_list_acc = split_trajs_list_by_prob(trajs_list_lg_b, n_probs)
        
        out_list = [] ## store output for each problems
        pick_traj_acc = []


        for i_pb in range(n_probs):
            trajs_list = trajs_list_acc[i_pb]

            ## traj picking and merging
            ## get unnormed numpy list, same format
            trajs_list_np_un = utils.get_np_trajs_list(trajs_list, do_unnorm=True, 
                                                    normalizer=self.normalizer)
            ## ranking of all the traj candiates based on the distance of ovlp parts
            s_idxs, dist_per_sam = utils.compute_ovlp_dist(trajs_list_np_un, 
                                                        self.diffusion_model.len_ovlp_cd)
            
            ## list, pick out the topn, from un-normed traj
            trajs_list_topn_np_un = utils.pick_top_n_trajs(trajs_list_np_un, s_idxs, self.top_n)
            ## np, un-normed, shape (B, tot_hzn, dim)
            trajs_list_topn_bl = self.tj_blder.blend_traj_lists(trajs_list_topn_np_un, do_unnorm=False)

            ## pick one traj to execute
            if self.pick_type == 'first':
                pick_traj = trajs_list_topn_bl[0]
            elif self.pick_type == 'rand':
                p_idx = np.random.randint(low=0, high=self.top_n)
                pick_traj = trajs_list_topn_bl[p_idx]
            else:
                raise NotImplementedError

            out = Stgl_Sml_Ev_Pred(pick_traj, trajs_list_topn_bl, trajs_list_np_un)

            out_list.append(out)
            pick_traj_acc.append(out.pick_traj)

        ##
        # pdb.set_trace()
        ## return a list of out and pick_traj
        return out_list, pick_traj_acc







    

    ### -------- Oct 20 For Stgl Quan Planning ------
    def gen_cond_stgl(self, g_cond, debug=False, b_s=1):
        """
        Jan 21: Default Version that Support Replan
        st_gl: *not normed*, np2d [2, ndim], e.g., [ [st], [end] ], [[2,1], [3,4]],
        b_s: batch_size, 10-20+
        """
        if self.n_comp == 1:
            return self.gen_1_cond_stgl(g_cond, b_s=b_s)
        
        hzn = self.diffusion_model.horizon
        o_dim = self.diffusion_model.observation_dim ## TODO: obs_dim only?
        c_shape = [b_s, hzn, o_dim] ## e.g.,(20,160,2)
        
        # pdb.set_trace() ## check format

        st_gl = g_cond['st_gl']
        st_gl = torch.tensor(self.normalizer.normalize(st_gl, 'observations'))
        
        ## shape: 2, n_probs, dim
        assert st_gl.ndim == 3 and st_gl.shape[0] == 2
        
        ## make sure return is not a view
        stgl_cond = {
            0: einops.repeat(st_gl[0,:,:], 'n_p d -> (n_p rr) d', rr=b_s).clone(),
            hzn-1: einops.repeat(st_gl[1,:,:], 'n_p d -> (n_p rr) d', rr=b_s).clone(),
        }

        cur_time = time.time()

        ## Run GPU Planning, x_dfu_all
        ## a list of len n_comp, elem: cuda tensor (B,sm_hzn,dim)
        if self.cp_infer_t_type == 'interleave': ## original our
            trajs_list = self.diffusion_model.comp_pred_p_loop_n(
                c_shape, stgl_cond, n_comp=self.n_comp, return_diffusion=False)
        
        elif self.cp_infer_t_type == 'same_t': ## Same t denoising, but not parallel
            trajs_list = self.diffusion_model.comp_pred_p_loop_n_same_t(
                c_shape, stgl_cond, n_comp=self.n_comp, return_diffusion=False)
            
        elif self.cp_infer_t_type == 'gsc': ## baseline
            trajs_list = self.diffusion_model.comp_pred_p_loop_n_GSC(
                c_shape, stgl_cond, n_comp=self.n_comp, return_diffusion=False)
        
        elif self.cp_infer_t_type == 'cdgs':
            trajs_list = self.diffusion_model.comp_pred_p_loop_n_CDGS(
                c_shape, stgl_cond, n_comp=self.n_comp, return_diffusion=False)
        
        elif self.cp_infer_t_type == 'same_t_p': ## Same t denoising and *parallel*
            trajs_list = self.diffusion_model.comp_pred_p_loop_n_same_t_parallel(
                c_shape, stgl_cond, n_comp=self.n_comp, return_diffusion=False)
        
        elif self.cp_infer_t_type == 'ar_back': ## backward autoregressive denosing
            trajs_list = self.diffusion_model.comp_pred_p_loop_n_ar_backward(
                c_shape, stgl_cond, n_comp=self.n_comp, return_diffusion=False)
        else:
            raise NotImplementedError
        
        self.ncp_pred_time_list.append( [self.n_comp,  time.time() - cur_time] ) ## unit: sec


        
        ## note that we can return a lof of stuff
        ## get unnormed numpy list, same format
        trajs_list_np_un = utils.get_np_trajs_list(trajs_list, do_unnorm=True, 
                                                   normalizer=self.normalizer)
        ## ranking of all the traj candiates based on the distance of ovlp parts
        s_idxs, dist_per_sam = utils.compute_ovlp_dist(trajs_list_np_un, 
                                                       self.diffusion_model.len_ovlp_cd)

        trajs_list_topn_bl, pick_traj = self.pick_and_blend_chunks(
            trajs_list=trajs_list,
            trajs_list_np_un=trajs_list_np_un,
            s_idxs=s_idxs,
            dist_per_sam=dist_per_sam,
        )

        ## pick one traj to execute
        if self.pick_type == 'rand':
            p_idx = np.random.randint(low=0, high=self.top_n)
            pick_traj = trajs_list_topn_bl[p_idx]

        out = Stgl_Sml_Ev_Pred(pick_traj, trajs_list_topn_bl, trajs_list_np_un)

        return out

    def pick_and_blend_chunks(self, trajs_list, trajs_list_np_un, s_idxs, dist_per_sam):
        if self.meta_method in ['baseline', 'ovlp_rank', 'cdgs']:
            self.last_stitch_info = dict(
                meta_method=self.meta_method,
                selected_sample_indices=tuple(int(v) for v in s_idxs[: self.top_n]),
                overlap_dist=[float(dist_per_sam[int(v)]) for v in s_idxs[: self.top_n]],
            )
            trajs_list_topn_np_un = utils.pick_top_n_trajs(
                trajs_list_np_un, s_idxs, self.top_n
            )
            trajs_list_topn_bl = self.tj_blder.blend_traj_lists(
                trajs_list_topn_np_un, do_unnorm=False
            )
            return trajs_list_topn_bl, trajs_list_topn_bl[0]

        if self.meta_method == 'local_repair_tree':
            density_scores = self.estimate_density_scores(trajs_list)
            sample_idx = int(s_idxs[0])
            chunk_list = [
                np.array(chunk_trajs[sample_idx], copy=True)
                for chunk_trajs in trajs_list_np_un
            ]
            density_list = density_scores[:, sample_idx].astype(np.float32).tolist()
            repair_out = self.search_local_repairs(
                chunk_list=chunk_list,
                density_list=density_list,
            )
            blended_traj = self.tj_blder.compose_single_traj(
                repair_out['chunk_list'],
                repair_out['seam_ops'],
                repair_out['density_list'],
            )
            trajs_list_topn_bl = np.repeat(
                blended_traj[None, :, :],
                repeats=self.top_n,
                axis=0,
            )
            self.last_stitch_info = dict(
                meta_method=self.meta_method,
                global_sample_idx=sample_idx,
                overlap_dist=float(dist_per_sam[sample_idx]),
                search_score=float(repair_out['score']),
                seam_ops=repair_out['seam_ops'],
                seam_logs=repair_out['seam_logs'],
                risky_seams=repair_out['risky_seams'],
            )
            return trajs_list_topn_bl, blended_traj

        if self.meta_method in ['global_density_guide', 'rcd']:
            density_scores = self.estimate_density_scores(trajs_list)
            shortlist_size = min(
                max(self.top_n, self.global_density_candidate_topk),
                len(s_idxs),
            )
            shortlist = [int(v) for v in s_idxs[:shortlist_size]]
            candidate_trajs = []
            candidate_payloads = []
            for sample_idx in shortlist:
                chunk_list = [
                    np.array(chunk_trajs[sample_idx], copy=True)
                    for chunk_trajs in trajs_list_np_un
                ]
                density_list = density_scores[:, sample_idx].astype(np.float32).tolist()
                seam_ops = ['exp'] * (len(chunk_list) - 1)
                blended = self.tj_blder.compose_single_traj(
                    chunk_list,
                    seam_ops,
                    density_list,
                )
                candidate_trajs.append(blended)
                candidate_payloads.append(
                    dict(
                        sample_idx=sample_idx,
                        blended_traj=blended,
                        seam_ops=seam_ops,
                        chunk_density=density_list,
                        overlap_dist=float(dist_per_sam[sample_idx]),
                    )
                )

            global_scores = self.estimate_global_density_np_batch(candidate_trajs)
            global_scores_norm = self.normalize_metric_rows(global_scores[None, :])[0]
            overlap_short = np.array(
                [payload['overlap_dist'] for payload in candidate_payloads],
                dtype=np.float32,
            )
            overlap_short_norm = self.normalize_metric_rows(overlap_short[None, :])[0]
            total_scores = -(
                global_scores_norm + self.global_overlap_weight * overlap_short_norm
            )

            ranked = sorted(
                [
                    dict(
                        score=float(total_scores[i_c]),
                        global_density=float(global_scores[i_c]),
                        **candidate_payloads[i_c],
                    )
                    for i_c in range(len(candidate_payloads))
                ],
                key=lambda item: item['score'],
                reverse=True,
            )
            top_results = ranked[: self.top_n]
            trajs_list_topn_bl = np.stack(
                [item['blended_traj'] for item in top_results],
                axis=0,
            )
            best = top_results[0]
            self.last_stitch_info = dict(
                meta_method=self.meta_method,
                cp_infer_t_type=self.cp_infer_t_type,
                search_score=float(best['score']),
                global_sample_idx=int(best['sample_idx']),
                overlap_dist=float(best['overlap_dist']),
                global_density=float(best['global_density']),
                chunk_density=best['chunk_density'],
                seam_ops=best['seam_ops'],
                top_sample_indices=[int(item['sample_idx']) for item in top_results],
                top_global_density=[float(item['global_density']) for item in top_results],
            )
            return trajs_list_topn_bl, best['blended_traj']

        if self.meta_method != 'density_tree':
            if self.meta_method == 'risk_gated_ops':
                density_scores = self.estimate_density_scores(trajs_list)
                sample_idx = int(s_idxs[0])
                chunk_list = [chunk_trajs[sample_idx] for chunk_trajs in trajs_list_np_un]
                density_list = density_scores[:, sample_idx]
                seam_logs = []
                seam_ops = []

                for i_c in range(len(chunk_list) - 1):
                    density_pair = (
                        float(density_list[i_c]),
                        float(density_list[i_c + 1]),
                    )
                    _, exp_metrics = self.tj_blder.blend_pair(
                        chunk_list[i_c],
                        chunk_list[i_c + 1],
                        blend_type='exp',
                        density_pair=density_pair,
                        return_metrics=True,
                    )
                    exp_cost = self.compute_seam_cost(exp_metrics)
                    best_op = 'exp'
                    best_cost = exp_cost
                    for op_name in ['cosine', 'smoothstep', 'min_jerk', 'density_gate']:
                        _, op_metrics = self.tj_blder.blend_pair(
                            chunk_list[i_c],
                            chunk_list[i_c + 1],
                            blend_type=op_name,
                            density_pair=density_pair,
                            return_metrics=True,
                        )
                        op_cost = self.compute_seam_cost(op_metrics)
                        if op_cost < best_cost:
                            best_cost = op_cost
                            best_op = op_name

                    if (
                        exp_cost > self.risk_threshold
                        and (exp_cost - best_cost) > self.switch_margin
                    ):
                        chosen_op = best_op
                    else:
                        chosen_op = 'exp'
                    seam_ops.append(chosen_op)
                    seam_logs.append(
                        dict(
                            seam=i_c,
                            exp_cost=float(exp_cost),
                            best_cost=float(best_cost),
                            chosen_op=chosen_op,
                            density_pair=density_pair,
                        )
                    )

                blended_traj = self.tj_blder.compose_single_traj(
                    chunk_list,
                    seam_ops,
                    density_list.tolist(),
                )
                trajs_list_topn_bl = np.repeat(
                    blended_traj[None, :, :],
                    repeats=self.top_n,
                    axis=0,
                )
                self.last_stitch_info = dict(
                    meta_method=self.meta_method,
                    global_sample_idx=sample_idx,
                    overlap_dist=float(dist_per_sam[sample_idx]),
                    seam_ops=seam_ops,
                    seam_logs=seam_logs,
                )
                return trajs_list_topn_bl, blended_traj

            if self.meta_method != 'density_tree_sameidx':
                raise NotImplementedError(self.meta_method)

            density_scores = self.estimate_density_scores(trajs_list)
            shortlist_size = min(
                max(self.top_n, self.tj_blder.search_chunk_pool * 2),
                len(s_idxs),
            )
            shortlist = [int(v) for v in s_idxs[:shortlist_size]]
            density_short = density_scores[:, shortlist]
            density_short_norm = self.normalize_metric_rows(density_short)
            dist_short = np.array(
                [float(dist_per_sam[s_idx]) for s_idx in shortlist], dtype=np.float32
            )
            dist_short_norm = self.normalize_metric_rows(dist_short[None, :])[0]

            candidate_results = []
            for local_idx, sample_idx in enumerate(shortlist):
                candidate_chunks = [
                    chunk_trajs[sample_idx : sample_idx + 1] for chunk_trajs in trajs_list_np_un
                ]
                candidate_density = density_short_norm[:, local_idx : local_idx + 1]
                search_out = self.tj_blder.search_blend_traj_lists(
                    trajs_list_np_un=candidate_chunks,
                    density_scores=candidate_density,
                )
                total_score = search_out.score - self.global_overlap_weight * float(
                    dist_short_norm[local_idx]
                )
                candidate_results.append(
                    dict(
                        score=float(total_score),
                        sample_idx=sample_idx,
                        blended_traj=search_out.blended_traj,
                        operators=search_out.operators,
                        diagnostics=search_out.diagnostics,
                        overlap_dist=float(dist_per_sam[sample_idx]),
                        density_scores=density_short[:, local_idx].tolist(),
                    )
                )

            candidate_results = sorted(
                candidate_results,
                key=lambda item: item['score'],
                reverse=True,
            )
            top_results = candidate_results[: self.top_n]
            trajs_list_topn_bl = np.stack(
                [item['blended_traj'] for item in top_results], axis=0
            )
            best = top_results[0]
            self.last_stitch_info = dict(
                meta_method=self.meta_method,
                search_score=best['score'],
                global_sample_idx=best['sample_idx'],
                operators=best['operators'],
                overlap_dist=best['overlap_dist'],
                density_scores=best['density_scores'],
                top_sample_indices=[item['sample_idx'] for item in top_results],
            )
            return trajs_list_topn_bl, best['blended_traj']

        density_scores = self.estimate_density_scores(trajs_list)
        candidate_chunks, candidate_density, candidate_globals = self.build_search_pools(
            trajs_list_np_un=trajs_list_np_un,
            density_scores=density_scores,
        )
        search_out = self.tj_blder.search_blend_traj_lists(
            trajs_list_np_un=candidate_chunks,
            density_scores=candidate_density,
        )
        global_chunk_indices = tuple(
            int(candidate_globals[i_c][search_out.chunk_indices[i_c]])
            for i_c in range(len(candidate_globals))
        )
        self.last_stitch_info = dict(
            meta_method=self.meta_method,
            search_score=float(search_out.score),
            chunk_indices=global_chunk_indices,
            operators=search_out.operators,
            diagnostics=search_out.diagnostics,
            density_scores=density_scores[:, list(global_chunk_indices)].diagonal().tolist(),
        )
        trajs_list_topn_bl = np.repeat(
            search_out.blended_traj[None, :, :],
            repeats=self.top_n,
            axis=0,
        )
        return trajs_list_topn_bl, search_out.blended_traj

    def estimate_density_scores(self, trajs_list):
        density_scores = []
        for chunk_trajs in trajs_list:
            d_proxy = self.diffusion_model.estimate_density_proxy(
                chunk_trajs,
                p_ratio=self.density_p_ratio,
                n_mc_samples=self.density_n_mc,
            )
            density_scores.append(utils.to_np(d_proxy))
        density_scores = np.stack(density_scores, axis=0)
        return density_scores

    def estimate_density_np_batch(self, chunk_list_np, focus_mask=None):
        if len(chunk_list_np) == 0:
            return np.zeros((0,), dtype=np.float32)
        chunk_norm = np.stack(
            [
                self.normalizer.normalize(chunk_np, 'observations')
                for chunk_np in chunk_list_np
            ],
            axis=0,
        )
        chunk_norm = torch.tensor(
            chunk_norm,
            device=self.device,
            dtype=torch.float32,
        )
        focus_mask_t = None
        if focus_mask is not None:
            focus_mask_t = torch.tensor(
                np.array(focus_mask, dtype=np.float32, copy=False),
                device=self.device,
                dtype=torch.float32,
            )
        d_proxy = self.diffusion_model.estimate_density_proxy(
            chunk_norm,
            p_ratio=self.density_p_ratio,
            n_mc_samples=self.density_n_mc,
            focus_mask=focus_mask_t,
        )
        return utils.to_np(d_proxy).astype(np.float32)

    def estimate_global_density_np_batch(self, traj_list_np):
        if len(traj_list_np) == 0:
            return np.zeros((0,), dtype=np.float32)
        traj_norm = np.stack(
            [
                self.normalizer.normalize(traj_np, 'observations')
                for traj_np in traj_list_np
            ],
            axis=0,
        )
        traj_norm = torch.tensor(
            traj_norm,
            device=self.device,
            dtype=torch.float32,
        )
        d_proxy, _ = self.diffusion_model.estimate_global_density_proxy(
            traj_norm,
            p_ratio=self.density_p_ratio,
            n_mc_samples=self.global_density_n_mc,
            proxy_type=self.global_density_proxy_type,
            overlap_weight=self.global_density_proxy_overlap_weight,
            window_beta=self.global_density_window_beta,
        )
        return utils.to_np(d_proxy).astype(np.float32)

    def build_local_focus_mask(self, batch_size, side):
        horizon = self.diffusion_model.horizon
        len_ovlp = self.diffusion_model.len_ovlp_cd
        focus_len = int(round(len_ovlp * self.local_focus_ratio))
        focus_len = max(2, min(len_ovlp, focus_len))
        focus_start = max(0, (len_ovlp - focus_len) // 2)
        focus_end = focus_start + focus_len

        mask = np.zeros((batch_size, horizon), dtype=np.float32)
        if side == 'left':
            offset = horizon - len_ovlp
        elif side == 'right':
            offset = 0
        else:
            raise ValueError(side)
        mask[:, offset + focus_start : offset + focus_end] = 1.0
        return mask

    def evaluate_repair_ops(self, chunk_list, density_list, seam_idx):
        left_chunk = chunk_list[seam_idx]
        right_chunk = chunk_list[seam_idx + 1]
        density_pair = (
            float(density_list[seam_idx]),
            float(density_list[seam_idx + 1]),
        )
        op_names = [
            'exp',
            'smoothstep',
            'density_gate',
            'min_jerk',
            'hold_bridge',
            'mode_patch',
        ]

        overlaps = []
        seam_metrics = []
        seam_costs = []
        left_mods = []
        right_mods = []
        keep_overlap = None

        for op_name in op_names:
            overlap, metrics = self.tj_blder.blend_pair(
                left_chunk,
                right_chunk,
                blend_type=op_name,
                density_pair=density_pair,
                return_metrics=True,
            )
            overlap = overlap.astype(np.float32)
            if keep_overlap is None:
                keep_overlap = overlap
            left_new = np.array(left_chunk, copy=True)
            right_new = np.array(right_chunk, copy=True)
            left_new[-self.diffusion_model.len_ovlp_cd :, :] = overlap
            right_new[: self.diffusion_model.len_ovlp_cd, :] = overlap

            overlaps.append(overlap)
            seam_metrics.append(metrics)
            seam_costs.append(self.compute_seam_cost(metrics))
            left_mods.append(left_new)
            right_mods.append(right_new)

        left_density = self.estimate_density_np_batch(left_mods)
        right_density = self.estimate_density_np_batch(right_mods)
        density_sums = left_density + right_density
        left_focus_mask = self.build_local_focus_mask(len(left_mods), side='left')
        right_focus_mask = self.build_local_focus_mask(len(right_mods), side='right')
        left_local_density = self.estimate_density_np_batch(
            left_mods,
            focus_mask=left_focus_mask,
        )
        right_local_density = self.estimate_density_np_batch(
            right_mods,
            focus_mask=right_focus_mask,
        )
        local_density_sums = left_local_density + right_local_density
        edit_sizes = np.array(
            [
                float(np.mean((overlap - keep_overlap) ** 2))
                for overlap in overlaps
            ],
            dtype=np.float32,
        )

        seam_costs = np.array(seam_costs, dtype=np.float32)
        density_sums = np.array(density_sums, dtype=np.float32)
        local_density_sums = np.array(local_density_sums, dtype=np.float32)
        cost_scale = float(np.std(seam_costs))
        density_scale = float(np.std(density_sums))
        local_density_scale = float(np.std(local_density_sums))
        edit_scale = float(np.std(edit_sizes))

        def scaled_delta(cur_val, new_val, scale):
            if scale < 1e-6:
                return 0.0
            return float((cur_val - new_val) / scale)

        results = []
        keep_cost = float(seam_costs[0])
        keep_density = float(density_sums[0])
        keep_local_density = float(local_density_sums[0])
        for i_op, op_name in enumerate(op_names):
            gain = 0.0
            if i_op > 0:
                gain += scaled_delta(keep_cost, float(seam_costs[i_op]), cost_scale)
                gain += self.tj_blder.search_density_weight * scaled_delta(
                    keep_density,
                    float(density_sums[i_op]),
                    density_scale,
                )
                cost_increase = float(seam_costs[i_op] - keep_cost)
                if cost_increase <= self.local_cost_guard:
                    gain += self.local_density_weight * scaled_delta(
                        keep_local_density,
                        float(local_density_sums[i_op]),
                        local_density_scale,
                    )
                if edit_scale >= 1e-6:
                    gain -= self.tj_blder.search_edit_weight * float(
                        edit_sizes[i_op] / edit_scale
                    )

            new_chunk_list = [np.array(chunk_np, copy=True) for chunk_np in chunk_list]
            new_chunk_list[seam_idx] = left_mods[i_op]
            new_chunk_list[seam_idx + 1] = right_mods[i_op]
            new_density_list = np.array(density_list, dtype=np.float32, copy=True)
            new_density_list[seam_idx] = left_density[i_op]
            new_density_list[seam_idx + 1] = right_density[i_op]

            results.append(
                dict(
                    op_name=op_name,
                    gain=float(gain),
                    chunk_list=new_chunk_list,
                    density_list=new_density_list.tolist(),
                    seam_cost=float(seam_costs[i_op]),
                    density_sum=float(density_sums[i_op]),
                    local_density_sum=float(local_density_sums[i_op]),
                    edit_mse=float(edit_sizes[i_op]),
                    metrics=seam_metrics[i_op],
                )
            )

        return dict(
            seam=seam_idx,
            keep_cost=keep_cost,
            keep_density=keep_density,
            keep_local_density=keep_local_density,
            candidates=results,
        )

    def search_local_repairs(self, chunk_list, density_list):
        base_seam_scores = []
        local_rank_scores = []
        for seam_idx in range(len(chunk_list) - 1):
            density_pair = (
                float(density_list[seam_idx]),
                float(density_list[seam_idx + 1]),
            )
            _, metrics = self.tj_blder.blend_pair(
                chunk_list[seam_idx],
                chunk_list[seam_idx + 1],
                blend_type='exp',
                density_pair=density_pair,
                return_metrics=True,
            )
            base_seam_scores.append(float(self.compute_seam_cost(metrics)))
            if self.local_rank_weight > 0.0:
                left_local = self.estimate_density_np_batch(
                    [chunk_list[seam_idx]],
                    focus_mask=self.build_local_focus_mask(1, side='left'),
                )[0]
                right_local = self.estimate_density_np_batch(
                    [chunk_list[seam_idx + 1]],
                    focus_mask=self.build_local_focus_mask(1, side='right'),
                )[0]
                local_rank_scores.append(float(left_local + right_local))
            else:
                local_rank_scores.append(0.0)

        rank_scores = np.array(base_seam_scores, dtype=np.float32)
        if self.local_rank_weight > 0.0 and len(rank_scores) > 1:
            rank_scores = self.normalize_metric_rows(rank_scores[None, :])[0]
            local_rank_scores = self.normalize_metric_rows(
                np.array(local_rank_scores, dtype=np.float32)[None, :]
            )[0]
            rank_scores = rank_scores + self.local_rank_weight * local_rank_scores

        seam_order = sorted(
            range(len(rank_scores)),
            key=lambda idx: rank_scores[idx],
            reverse=True,
        )
        seam_order = [
            int(idx)
            for idx in seam_order
            if base_seam_scores[idx] >= self.risk_threshold
        ][: self.repair_top_k]

        beam = [
            dict(
                score=0.0,
                chunk_list=[np.array(chunk_np, copy=True) for chunk_np in chunk_list],
                density_list=np.array(density_list, dtype=np.float32).tolist(),
                seam_ops=['exp'] * (len(chunk_list) - 1),
                seam_logs=[],
            )
        ]

        for seam_idx in seam_order:
            next_beam = []
            for node in beam:
                seam_eval = self.evaluate_repair_ops(
                    chunk_list=node['chunk_list'],
                    density_list=node['density_list'],
                    seam_idx=seam_idx,
                )
                for cand in seam_eval['candidates']:
                    if cand['op_name'] != 'exp' and cand['gain'] <= self.switch_margin:
                        continue
                    new_ops = list(node['seam_ops'])
                    new_ops[seam_idx] = cand['op_name']
                    new_logs = list(node['seam_logs'])
                    new_logs.append(
                        dict(
                            seam=seam_idx,
                            keep_cost=float(seam_eval['keep_cost']),
                            keep_density=float(seam_eval['keep_density']),
                            keep_local_density=float(seam_eval['keep_local_density']),
                            chosen_op=cand['op_name'],
                            gain=float(cand['gain']),
                            new_cost=float(cand['seam_cost']),
                            new_density=float(cand['density_sum']),
                            new_local_density=float(cand['local_density_sum']),
                            edit_mse=float(cand['edit_mse']),
                        )
                    )
                    next_beam.append(
                        dict(
                            score=float(node['score'] + cand['gain']),
                            chunk_list=cand['chunk_list'],
                            density_list=cand['density_list'],
                            seam_ops=new_ops,
                            seam_logs=new_logs,
                        )
                    )

            if len(next_beam) == 0:
                break
            next_beam = sorted(
                next_beam,
                key=lambda item: item['score'],
                reverse=True,
            )
            beam = next_beam[: self.tj_blder.search_beam_width]

        best = beam[0]
        return dict(
            score=float(best['score']),
            chunk_list=best['chunk_list'],
            density_list=best['density_list'],
            seam_ops=best['seam_ops'],
            seam_logs=best['seam_logs'],
            risky_seams=[int(v) for v in seam_order],
        )

    def build_search_pools(self, trajs_list_np_un, density_scores):
        candidate_chunks = []
        candidate_density = []
        candidate_globals = []
        pool_size = self.tj_blder.search_chunk_pool
        for i_c, chunk_trajs in enumerate(trajs_list_np_un):
            ranked = np.argsort(density_scores[i_c])[: min(pool_size, len(chunk_trajs))]
            candidate_globals.append(ranked)
            candidate_chunks.append(chunk_trajs[ranked])
            candidate_density.append(density_scores[i_c][ranked])
        candidate_density = np.stack(candidate_density, axis=0)
        candidate_density = self.normalize_metric_rows(candidate_density)
        return candidate_chunks, candidate_density, candidate_globals

    def normalize_metric_rows(self, vals):
        vals = np.array(vals, dtype=np.float32, copy=True)
        if vals.ndim == 1:
            vals = vals[None, :]
        for i_r in range(vals.shape[0]):
            row = vals[i_r]
            if len(row) <= 1:
                vals[i_r] = 0.0
                continue
            row = row - np.min(row)
            row = row / (np.std(row) + 1e-6)
            vals[i_r] = row
        return vals

    def compute_seam_cost(self, seam_metrics):
        return (
            self.tj_blder.search_overlap_weight * seam_metrics['fit_mse']
            + self.tj_blder.search_commit_weight * seam_metrics['center_commit_mse']
            + self.tj_blder.search_vel_weight * seam_metrics['vel_mse']
            + self.tj_blder.search_acc_weight * seam_metrics['acc_mse']
            + self.tj_blder.search_rough_weight * seam_metrics['rough_mse']
        )
    


    def gen_1_cond_stgl(self, g_cond, debug=False, b_s=1):
        """
        use start goal as condition, do not compose model, just like vanilla DD

        st_gl: *not normed*, np2d [2, ndim], e.g., [ [st], [end] ], [[2,1], [3,4]],
        b_s: batch_size, 10-20+
        """
        
        hzn = self.diffusion_model.horizon
        o_dim = self.diffusion_model.observation_dim
        c_shape = [b_s, hzn, o_dim] ## e.g.,(20,160,2)
        
        # pdb.set_trace() ## check format

        st_gl = g_cond['st_gl']
        st_gl = torch.tensor(self.normalizer.normalize(st_gl, 'observations'))
        
        n_probs = st_gl.shape[1]
        ## shape: 2, n_probs, dim, now only support planning for one problem
        assert st_gl.ndim == 3 and st_gl.shape[0] == 2 and n_probs == 1

        # pdb.set_trace()

        stgl_cond = {
            0: einops.repeat(st_gl[0,:,:], 'n_p d -> (n_p rr) d', rr=b_s).clone(),
            hzn-1: einops.repeat(st_gl[1,:,:], 'n_p d -> (n_p rr) d', rr=b_s).clone(),
        }

        g_cond_1 = {}
        g_cond_1['do_cond'] = 'both_stgl'
        g_cond_1['stgl_cond'] = stgl_cond
        ## uesless placeholder
        g_cond_1['t_type'] = '0'
        g_cond_1['traj_full'] = np.random.random( size=(stgl_cond[0].shape[0],) )

        pred_trajs = self.diffusion_model.conditional_sample(g_cond=g_cond_1)

        pred_trajs = apply_conditioning(pred_trajs, stgl_cond, 0)



        pred_trajs = utils.to_np(pred_trajs)
        pred_trajs_un = self.normalizer.unnormalize(pred_trajs, 'observations')

        out = Stgl_Sml_Ev_Pred(pred_trajs_un[0], 
                               pred_trajs_un, 
                               pred_trajs_un)
        ##
        # pdb.set_trace()

        return out





    




    def _format_g_cond(self, g_cond, batch_size):
        
        traj_f = g_cond['traj_full'] 
        ## normalize the traj
        traj_f =  self.normalizer.normalize(traj_f, 'observations')
        traj_f = torch.tensor(traj_f, dtype=torch.float32, device='cuda:0')

        traj_f = einops.repeat(traj_f, 'b h d -> (repeat b) h d', repeat=batch_size)
        
        g_cond['traj_full'] = traj_f
        
        return g_cond


    def gen_cond(self, g_cond, debug=False, batch_size=1):
        '''conditional sampling
        conditioned on start and end chunks, just for sanity test
        '''

        g_cond = self._format_g_cond(g_cond, batch_size)

        
        sample = self.diffusion_model.conditional_sample(g_cond)

        
        actions = np.zeros(shape=(*sample.shape[0:2], self.action_dim))
        sample = utils.to_np(sample)
        actions = self.normalizer.unnormalize(actions, 'actions')
        # actions = np.tanh(actions)
        
        ## extract first action
        action = actions[0, 0]

        # pdb.set_trace()

        # if debug:
        normed_observations = sample[:, :, 0:]
        observations = self.normalizer.unnormalize(normed_observations, 'observations')

        trajectories = Trajectories_invdyn(actions, observations)
        return action, trajectories
        




    

class Stgl_Sml_Ev_Pred:
    def __init__(self, pick_traj, 
                 trajs_list_topn_bl, trajs_list_np_un) -> None:
        '''
        pick_traj: np2d unnormed (tot_hzn,dim), the traj to follow
        trajs_list_topn_bl: np3d unnormed (B,tot_hzn,dim), all topn


        '''
        self.pick_traj = pick_traj
        self.trajs_list_topn_bl = trajs_list_topn_bl
        self.trajs_list_np_un = trajs_list_np_un
