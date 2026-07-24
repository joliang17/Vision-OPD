set -euo pipefail
OPSD_ROOT=/mnt/bn/tns-algo-video-vlm-ruby/yijunliang/project/opsd
mkdir -p "${OPSD_ROOT}/Vision-OPD/scripts/smoketest_out"
date -u > "${OPSD_ROOT}/Vision-OPD/scripts/smoketest_out/marker1.txt"
echo "cwd=$(pwd)" >> "${OPSD_ROOT}/Vision-OPD/scripts/smoketest_out/marker1.txt"
nvidia-smi >> "${OPSD_ROOT}/Vision-OPD/scripts/smoketest_out/marker1.txt" 2>&1 || echo "nvidia-smi failed" >> "${OPSD_ROOT}/Vision-OPD/scripts/smoketest_out/marker1.txt"
cd "${OPSD_ROOT}/VLMEvalKit"
pip3 install -r requirements_arnold.txt >> "${OPSD_ROOT}/Vision-OPD/scripts/smoketest_out/marker1.txt" 2>&1
echo "pip_install_done" >> "${OPSD_ROOT}/Vision-OPD/scripts/smoketest_out/marker1.txt"
sleep 30
echo "smoketest survived 30s sleep" >> "${OPSD_ROOT}/Vision-OPD/scripts/smoketest_out/marker1.txt"
