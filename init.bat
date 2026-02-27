@REM ------------------------------------------------------------------------------------------
@REM network proxy
@REM 

set https_proxy=http://proxy-dmz.intel.com:912
set http_proxy=http://proxy-dmz.intel.com:911

@REM set ftp_proxy=http://child-prc.intel.com:913/
@REM set "HF_ENDPOINT=https://hf-mirror.com"

set no_proxy=localhost,127.0.0.0/8,::1,https://github.com/*


@REM ------------------------------------------------------------------------------------------
@REM github account
@REM 
git config --global user.email "cecilia.peng@intel.com"
git config --global user.name "ceciliapeng2011"




@REM ------------------------------------------------------------------------------------------
@REM global environments
@REM 
set HOME=c:\ceciliapeng

@REM genai build
set OpenVINO_DIR=%HOME%\openvino\release_install\runtime
set TBB_DIR=%HOME%\openvino\temp\Windows_AMD64\tbb\bin

@REM ov env
set PYTHONPATH=%HOME%\openvino\release_install\python;%HOME%\openvino\tools\ovc;%HOME%\openvino\bin\intel64\Release\python
set OPENVINO_LIB_PATHS=%HOME%\openvino\release_install\runtime\bin\intel64\Release\;%HOME%\openvino\temp\Windows_AMD64\tbb\bin;%HOME%\openvino\bin\intel64\Release\
set PATH=%HOME%\openvino\tools\ovc;%HOME%\openvino\bin\intel64\Release;%HOME%\openvino\temp\Windows_AMD64\tbb\bin;%PATH%

@REM for aboutSHW build with onednn
set OV_BUILD_PATH=%HOME%\openvino\build

@REM CMC
set CM_FE_DIR=%HOME%\ComputeSDK_Windows_internal_2025_WW41\compiler\bin


%HOME%\openvino.venv\Scripts\activate
cd %HOME%

