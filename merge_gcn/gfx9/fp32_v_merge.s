.include "gpr_alloc.inc"

.hsa_code_object_version 2,1
.hsa_code_object_isa

.GPR_ALLOC_BEGIN
    kernarg = 0
    gid_x = 2
    gsize = 128
    .SGPR_ALLOC_FROM 5
    .SGPR_ALLOC tmp
    .SGPR_ALLOC stmp, 2
    .SGPR_ALLOC base_in, 2
    .SGPR_ALLOC base_out, 2
    .SGPR_ALLOC s_exec, 2
    .SGPR_ALLOC bin_exec, 2
    .SGPR_ALLOC length
    .SGPR_ALLOC cur_len
    .VGPR_ALLOC_FROM 0
    .VGPR_ALLOC tid
    .VGPR_ALLOC voffset
    .VGPR_ALLOC vaddr, 2
    .VGPR_ALLOC vin_addr, 2
    .VGPR_ALLOC vout_addr, 2
    .VGPR_ALLOC in
    .VGPR_ALLOC out
    .VGPR_ALLOC a_start
    .VGPR_ALLOC a_end
    .VGPR_ALLOC b_start
    .VGPR_ALLOC b_end
    .VGPR_ALLOC vtmp
    .VGPR_ALLOC a_pos
    .VGPR_ALLOC l
    .VGPR_ALLOC r
    .VGPR_ALLOC m
    .VGPR_ALLOC b
.GPR_ALLOC_END


.text
.p2align 8
.amdgpu_hsa_kernel hello_world

hello_world:

    .amd_kernel_code_t
        is_ptr64 = 1
        enable_sgpr_kernarg_segment_ptr = 1
        enable_sgpr_workgroup_id_x = 1
        kernarg_segment_byte_size = 24
        compute_pgm_rsrc2_user_sgpr = 2
        granulated_workitem_vgpr_count = .AUTO_VGPR_GRANULATED_COUNT
        granulated_wavefront_sgpr_count = .AUTO_SGPR_GRANULATED_COUNT
        wavefront_sgpr_count = .AUTO_SGPR_COUNT
        workitem_vgpr_count = .AUTO_VGPR_COUNT
    .end_amd_kernel_code_t

  // read kernel arguments:
  // s[base_in1:base_in+1] = *in
  // s[base_out:base_out+1] = *out
  s_load_dwordx2        s[base_in:base_in+1], s[kernarg:kernarg+1], 0x00
  s_load_dwordx2        s[base_out:base_out+1], s[kernarg:kernarg+1], 0x08
  s_load_dwordx2        s[length:cur_len], s[kernarg:kernarg+1], 0x10
  s_waitcnt             0
  // group offset (group size 64)
  s_mul_i32             s[tmp], s[gid_x], gsize
  v_add_u32             v[tid], v[tid], s[tmp]
  .GPR_REUSE tid, gid
  // if (n <= gid) return
  v_cmp_gt_i32          vcc, s[length], v[gid]
  s_and_saveexec_b64    s[s_exec:s_exec+1], vcc
  //  unsigned a_start = global_id / (len << 1) * (len << 1);
  s_lshl_b32            s[tmp], s[cur_len], 1 
  v_cvt_f32_u32         v[a_start], s[tmp]
  v_rcp_iflag_f32       v[a_start], v[a_start]
  v_cvt_f32_u32         v[vtmp], v[gid]
  v_mul_f32             v[a_start], v[a_start], v[vtmp]
  v_cvt_u32_f32         v[a_start], v[a_start]
  v_mul_i32_i24         v[a_start], v[a_start], s[tmp]

  //unsigned a_end = min(a_start + len, n);
  v_add_u32             v[a_end], v[a_start], s[cur_len]
  v_min_u32             v[a_end], v[a_end], s[length]

  // unsigned b_start = a_end;
  v_mov_b32             v[b_start], v[a_end]
  // unsigned b_end = min(b_start + len, n);
  v_add_u32             v[b_end], v[b_start], s[cur_len]
  v_min_u32             v[b_end], v[b_end], s[length]
  // if (b_start >= n) {  as[global_id] = as_out[global_id]; return;    }
  
  v_cmp_ge_u32          vcc, v[b_start], s[length]
  s_andn1_saveexec_b64  s[s_exec:s_exec+1], vcc
  s_xor_b64             exec, vcc, exec           
  s_mov_b64             s[s_exec:s_exec+1], exec
  v_sub_u32             v[a_pos], v[gid], v[a_start]
  v_sub_i32             v[vtmp], v[b_end], v[b_start]
  v_sub_i32             v[l], v[a_pos], v[vtmp]
  v_max_i32             v[l], v[l], 0
  v_sub_i32             v[l], v[l], 1
  v_min_i32             v[r], s[cur_len], v[a_pos]
  bin_search_loop:
    v_add_i32                v[vtmp], v[l], 1
    v_cmp_gt_i32             vcc, v[r], v[vtmp]
    s_and_saveexec_b64       s[bin_exec:bin_exec+1], vcc
    v_add_i32                v[m], v[l], v[r]
    v_lshrrev_b32            v[m], 1, v[m]
    v_add_u32                v[voffset], v[a_start], v[m]
    v_lshlrev_b32            v[voffset], 2, v[voffset]
    v_add_co_u32			 v[vaddr], vcc, s[base_in], v[voffset]
    v_mov_b32				 v[vaddr+1], s[base_in+1]
    v_addc_co_u32			 v[vaddr+1], vcc, v[vaddr+1], 0, vcc
    flat_load_dword		     v[in], v[vaddr:vaddr+1]
    v_add_u32                v[voffset], v[b_start], v[a_pos]
    v_sub_u32                v[voffset], v[voffset], v[m]
    v_sub_u32                v[voffset], v[voffset], 1
    v_lshlrev_b32            v[voffset], 2, v[voffset]
    v_add_co_u32			 v[vaddr], vcc, s[base_in], v[voffset]
    v_mov_b32				 v[vaddr+1], s[base_in+1]
    v_addc_co_u32			 v[vaddr+1], vcc, v[vaddr+1], 0, vcc
    flat_load_dword		     v[out], v[vaddr:vaddr+1]
    s_waitcnt                0

    v_cmp_le_f32             vcc, v[in], v[out]
    s_and_saveexec_b64       s[bin_exec:bin_exec+1], vcc
    v_mov_b32                v[l], v[m]
    s_andn2_b64              exec, s[bin_exec:bin_exec+1], vcc
    v_mov_b32                v[r], v[m]
    s_mov_b64                exec, s[bin_exec:bin_exec+1]
    s_cbranch_execnz         bin_search_loop
  .GPR_REUSE vtmp, a
  s_mov_b64                exec, s[s_exec:s_exec+1]
  v_add_u32 v[a], v[a_start], v[r]
  v_add_u32 v[b], v[b_start], v[a_pos]
  v_sub_u32 v[b], v[b], v[r]

  v_lshlrev_b32            v[voffset], 2, v[a]
  v_add_co_u32			   v[vaddr], vcc, s[base_in], v[voffset]
  v_mov_b32				   v[vaddr+1], s[base_in+1]
  v_addc_co_u32			   v[vaddr+1], vcc, v[vaddr+1], 0, vcc
  flat_load_dword		   v[in], v[vaddr:vaddr+1]
  v_lshlrev_b32            v[voffset], 2, v[b]
  v_add_co_u32			   v[vaddr], vcc, s[base_in], v[voffset]
  v_mov_b32				   v[vaddr+1], s[base_in+1]
  v_addc_co_u32			   v[vaddr+1], vcc, v[vaddr+1], 0, vcc
  flat_load_dword		   v[out], v[vaddr:vaddr+1]
  s_waitcnt                0
  v_cmp_ge_u32          s[stmp:stmp+1], v[b], v[b_end]
  v_cmp_le_u32          vcc, v[in], v[out]
  s_or_b64              s[stmp:stmp+1], vcc,  s[stmp:stmp+1]
  v_cmp_lt_u32          vcc, v[a], v[a_end]
  s_and_b64             s[stmp:stmp+1], vcc,  s[stmp:stmp+1]
  s_and_saveexec_b64    s[s_exec:s_exec+1], s[stmp:stmp+1]
  v_mov_b32             v[out], v[in]
  s_mov_b64             exec, s[s_exec:s_exec+1]

  v_lshlrev_b32	        v[voffset], 2, v[gid]
  v_add_co_u32			v[vaddr], vcc, s[base_out], v[voffset]
  v_mov_b32				v[vaddr+1], s[base_out+1]
  v_addc_co_u32			v[vaddr+1], vcc, v[vaddr+1], 0, vcc
  flat_store_dword		v[vaddr:vaddr+1], v[out]
  s_endpgm