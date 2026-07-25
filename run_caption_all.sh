#!/bin/bash
# ============================================================
# Qwen3-VL-8B Caption 批量推理启动脚本
# 2×H200 + 4×4090，每台机器使用独立 subset 文件
# ============================================================
#
# 数据分布 (仅处理训练数据中未覆盖的 34,855 条):
#   H200 机器 0:  5,810 条
#   H200 机器 1:  5,810 条
#   4090 机器 0:  5,810 条
#   4090 机器 1:  5,810 条
#   4090 机器 2:  5,810 条
#   4090 机器 3:  5,805 条
# ============================================================
# 用法:
#   bash run_caption_all.sh h200 0
#   bash run_caption_all.sh h200 1
#   bash run_caption_all.sh 4090 0
#   bash run_caption_all.sh 4090 1
#   bash run_caption_all.sh 4090 2
#   bash run_caption_all.sh 4090 3
# ============================================================
# 全部完成后，在任意一台执行合并:
#   cd /data/phd/kousiqi/zhitao && python3 qwenvl_rewrite.py --merge-only \
#       --tags h200_0 h200_1 4090_m0 4090_m1 4090_m2 4090_m3
# ============================================================

set -e

CLUSTER="${1}"
MACHINE_ID="${2}"

if [ -z "${CLUSTER}" ] || [ -z "${MACHINE_ID}" ]; then
    echo "用法:"
    echo "  bash run_caption_all.sh h200 0"
    echo "  bash run_caption_all.sh h200 1"
    echo "  bash run_caption_all.sh 4090 0"
    echo "  bash run_caption_all.sh 4090 1"
    echo "  bash run_caption_all.sh 4090 2"
    echo "  bash run_caption_all.sh 4090 3"
    exit 1
fi

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
NUM_GPUS=8

# ============================================================
# 每台机器使用独立的 subset 文件
# ============================================================
if [ "${CLUSTER}" = "h200" ]; then
    case "${MACHINE_ID}" in
        0) INPUT_FILE="${SCRIPT_DIR}/metadata_caption_h200_h200_0.json"; TAG="h200_0" ;;
        1) INPUT_FILE="${SCRIPT_DIR}/metadata_caption_h200_h200_1.json"; TAG="h200_1" ;;
        *) echo "错误: H200 机器编号需为 0 或 1"; exit 1 ;;
    esac
elif [ "${CLUSTER}" = "4090" ]; then
    case "${MACHINE_ID}" in
        0) INPUT_FILE="${SCRIPT_DIR}/metadata_caption_4090_m0.json"; TAG="4090_m0" ;;
        1) INPUT_FILE="${SCRIPT_DIR}/metadata_caption_4090_m1.json"; TAG="4090_m1" ;;
        2) INPUT_FILE="${SCRIPT_DIR}/metadata_caption_4090_m2.json"; TAG="4090_m2" ;;
        3) INPUT_FILE="${SCRIPT_DIR}/metadata_caption_4090_m3.json"; TAG="4090_m3" ;;
        *) echo "错误: 4090 机器编号需为 0-3"; exit 1 ;;
    esac
else
    echo "错误: 第一个参数需为 'h200' 或 '4090'"
    exit 1
fi

echo "============================================"
echo "Qwen3-VL-8B Caption 批量推理"
echo "  集群:      ${CLUSTER}"
echo "  机器编号:  ${MACHINE_ID}"
echo "  标签:      ${TAG}"
echo "  输入文件:  ${INPUT_FILE}"
echo "  GPU 数:    ${NUM_GPUS}"
echo "  中断后重跑相同命令即可断点续跑"
echo "============================================"
echo ""

cd "${SCRIPT_DIR}"

python3 qwenvl_rewrite.py \
    --input "${INPUT_FILE}" \
    --tag "${TAG}" \
    --num-gpus "${NUM_GPUS}" \
    --model-path "${SCRIPT_DIR}/models/Qwen3-VL-8B-Instruct"

RET=$?
echo ""
if [ ${RET} -eq 0 ]; then
    echo "✅ ${TAG} 完成!"
    echo ""
    echo "全部完成后，在任意一台执行合并:"
    echo "  cd ${SCRIPT_DIR} && python3 qwenvl_rewrite.py --merge-only --tags h200_0 h200_1 4090_m0 4090_m1 4090_m2 4090_m3"
else
    echo "❌ 异常退出 (exit code: ${RET})，重新运行即可断点续跑"
fi
