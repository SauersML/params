//! NVRTC compilation of the checked interval kernels (`fixed_logit_interval.cu`): gam-gpu's shared
//! device-keyed options with flush-to-zero off, contraction off, and IEEE-rounded division and
//! square root. No other kernel compiles with these options.

use cudarc::driver::{CudaContext, CudaModule};
use cudarc::nvrtc::{CompileOptions, compile_ptx_with_opts};
use gam_gpu::device_cache::nvrtc_compile_options;
use gam_gpu::driver::{CudarcLibrary, require_cudarc_library};
use gam_gpu::gpu_error::{GpuError, GpuResultExt};
use std::sync::{Arc, OnceLock};

/// A process-wide module compiled once with the checked options.
#[derive(Default)]
pub(crate) struct CheckedModuleCache {
    module: OnceLock<Arc<CudaModule>>,
}

impl CheckedModuleCache {
    pub(crate) const fn new() -> Self {
        Self { module: OnceLock::new() }
    }

    /// Compile `source` with the checked options and load it on `ctx` the first time; return the
    /// cached module on every later call.
    pub(crate) fn get_or_compile(&self, ctx: &Arc<CudaContext>, source: &str) -> Result<&Arc<CudaModule>, GpuError> {
        if let Some(existing) = self.module.get() {
            return Ok(existing);
        }
        require_cudarc_library(CudarcLibrary::Nvrtc)?;
        let ptx = compile_ptx_with_opts(source, options()?).gpu_ctx("checked interval NVRTC compile")?;
        let module = ctx.load_module(ptx).gpu_ctx("checked interval module load")?;
        if self.module.set(module).is_err() {
            log::trace!("concurrent checked interval module compile already populated cache");
        }
        Ok(self.module.get().expect("module populated after checked compilation"))
    }
}

fn options() -> Result<CompileOptions, GpuError> {
    let mut opts = nvrtc_compile_options()?;
    opts.ftz = Some(false);
    opts.fmad = Some(false);
    // cudarc emits no fastmath flag for Some(false); record the actual flags separately from this
    // explicitly disabled policy.
    opts.use_fast_math = Some(false);
    opts.prec_div = Some(true);
    opts.prec_sqrt = Some(true);
    Ok(opts)
}

/// The loaded NVRTC version and the numerical and architecture flags the checked path compiles
/// with. Querying this does not compile anything.
pub(crate) fn compiler_info() -> Result<(i32, i32, Vec<String>), GpuError> {
    require_cudarc_library(CudarcLibrary::Nvrtc)?;
    let (mut major, mut minor) = (0, 0);
    // SAFETY: both pointers address live writable stack integers; NVRTC is loaded above and the
    // return status is checked before using outputs.
    unsafe { cudarc::nvrtc::sys::nvrtcVersion(&mut major, &mut minor) }
        .result()
        .gpu_ctx("checked interval NVRTC version")?;
    let options = options()?;
    let mut flags = vec!["--ftz=false".into(), "--fmad=false".into(), "--prec-div=true".into(), "--prec-sqrt=true".into()];
    if let Some(arch) = options.arch {
        flags.push(format!("--gpu-architecture={arch}"));
    }
    Ok((major, minor, flags))
}
