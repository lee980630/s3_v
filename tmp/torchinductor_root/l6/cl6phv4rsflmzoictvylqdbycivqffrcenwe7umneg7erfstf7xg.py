
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.reduction(
    size_hints={'x': 8192, 'r': 262144},
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*bf16', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'in_ptr4': '*fp32', 'out_ptr1': '*bf16', 'ks0': 'i32', 'xnumel': 'i32', 'rnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=108, cc=80, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4, 5), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused__softmax__softmax_backward_data__to_copy_add_exp_mul_sub_0', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 6, 'num_reduction': 1, 'backend_hash': '47F64A9F6A85E0BCC1AF063FAD76D919A3624F50FDDB2C9704DA0603639DC738', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False}
)
@triton.jit
def triton_red_fused__softmax__softmax_backward_data__to_copy_add_exp_mul_sub_0(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr1, ks0, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (x0), xmask, eviction_policy='evict_last')
    tmp5 = tl.load(in_ptr2 + (x0), xmask, eviction_policy='evict_last')
    tmp8 = tl.load(in_ptr3 + (x0), xmask, eviction_policy='evict_last')
    _tmp12 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r1 = rindex
        tmp2 = tl.load(in_ptr1 + (r1 + ks0*x0), rmask & xmask, eviction_policy='evict_last', other=0.0).to(tl.float32)
        tmp1 = -tmp0
        tmp3 = tmp2.to(tl.float32)
        tmp4 = tmp1 * tmp3
        tmp6 = tmp3 - tmp5
        tmp7 = tl_math.exp(tmp6)
        tmp9 = tmp7 / tmp8
        tmp10 = tmp4 * tmp9
        tmp11 = tl.broadcast_to(tmp10, [XBLOCK, RBLOCK])
        tmp13 = _tmp12 + tmp11
        _tmp12 = tl.where(rmask & xmask, tmp13, _tmp12)
    tmp12 = tl.sum(_tmp12, 1)[:, None]
    tmp22 = tl.load(in_ptr4 + (x0), xmask, eviction_policy='evict_last')
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r1 = rindex
        tmp15 = tl.load(in_ptr1 + (r1 + ks0*x0), rmask & xmask, eviction_policy='evict_first', other=0.0).to(tl.float32)
        tmp14 = -tmp0
        tmp16 = tmp15.to(tl.float32)
        tmp17 = tmp16 - tmp5
        tmp18 = tl_math.exp(tmp17)
        tmp19 = tmp18 / tmp8
        tmp20 = tmp14 * tmp19
        tmp21 = tmp20.to(tl.float32)
        tmp23 = tmp16 - tmp22
        tmp24 = tl_math.exp(tmp23)
        tmp25 = tmp0 * tmp24
        tmp26 = tmp25.to(tl.float32)
        tmp27 = tmp21 + tmp26
        tmp28 = -tmp19
        tmp29 = tmp14 * tmp16
        tmp30 = tmp29 * tmp19
        tmp31 = libdevice.fma(tmp28, tmp12, tmp30)
        tmp32 = tmp31.to(tl.float32)
        tmp33 = tmp27 + tmp32
        tl.store(out_ptr1 + (r1 + ks0*x0), tmp33, rmask & xmask)
