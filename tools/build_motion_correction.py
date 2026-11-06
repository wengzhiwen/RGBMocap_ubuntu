"""Build the pinned Kimodo CPU backend; users receive the built module."""
from pathlib import Path
import argparse
import os
import shutil
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--deps', type=Path, default=ROOT/'.build-inputs/native-deps')
    parser.add_argument('--build-dir',type=Path,default=ROOT/'build/motion-correction')
    parser.add_argument('--output-dir',type=Path,default=ROOT/'third_party/KimodoMotionCorrection/bin')
    parser.add_argument('--python',type=Path,default=Path(sys.executable),help='Interpreter whose ABI the module targets; use .venv/bin/python on Linux.')
    args=parser.parse_args()
    cmake=shutil.which('cmake')
    if not cmake and os.name=='nt':
        candidates=list(Path('C:/Program Files (x86)/Microsoft Visual Studio').glob('*/BuildTools/Common7/IDE/CommonExtensions/Microsoft/CMake/CMake/bin/cmake.exe'))
        if candidates:cmake=str(candidates[-1])
    if not cmake:raise RuntimeError('Install CMake for the developer build')
    source=ROOT/'third_party/KimodoMotionCorrection'
    build=args.build_dir.resolve()
    build.mkdir(parents=True,exist_ok=True)
    if os.name=='nt':
        command=[cmake,'-S',str(source),'-B',str(build),'-G','Visual Studio 17 2022','-A','x64',
                 '-DPython3_EXECUTABLE='+sys.executable,'-DCMAKE_EXPORT_NO_PACKAGE_REGISTRY=ON',
                 '-DCMAKE_CXX_FLAGS=/utf-8 /EHsc /wd4127 /wd4267']
    else:
        # Do not resolve the interpreter: CMake must see the venv, not its base Python.
        command=[cmake,'-S',str(source),'-B',str(build),'-DCMAKE_BUILD_TYPE=Release',
                 '-DPython3_EXECUTABLE='+str(args.python),'-DCMAKE_EXPORT_NO_PACKAGE_REGISTRY=ON']
    if (args.deps/'pybind11/share/cmake/pybind11').exists():
        command+=['-Dpybind11_DIR='+str((args.deps/'pybind11/share/cmake/pybind11').resolve())]
    if (args.deps/'sse2neon').exists():
        command+=['-DFETCHCONTENT_SOURCE_DIR_SSE2NEON='+str((args.deps/'sse2neon').resolve())]
    if (args.deps/'eigen-3.4.0').exists():
        command+=['-DFETCHCONTENT_SOURCE_DIR_EIGEN='+str((args.deps/'eigen-3.4.0').resolve())]
    with (build/'build.log').open('w',encoding='utf-8') as log:
        subprocess.run(command,check=True,stdout=log,stderr=subprocess.STDOUT)
        subprocess.run([cmake,'--build',str(build),'--config','Release','--target','_motion_correction','--parallel','2'],check=True,stdout=log,stderr=subprocess.STDOUT)
    target=args.output_dir.resolve();target.mkdir(parents=True,exist_ok=True)
    for file in (build/'Release').glob('_motion_correction*.pyd') if os.name=='nt' else build.glob('_motion_correction*.so'):
        shutil.copy2(file,target/file.name)
        print(target/file.name)

if __name__=='__main__':main()
