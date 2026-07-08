#!/bin/bash
# ============================================================
# 一键启动：Qwen3-VL-8B Caption 批量推理
# 使用 transformers 直接加载模型，无需 vLLM
#
# 2×H200 + 4×4090 混合集群版
# ============================================================
#
# 数据分布 (总共 1,143,771 条):
#   H200 机器 0:        [     0 ~ 285943)   285,943 条  (~25%)
#   H200 机器 1:        [285943 ~ 571886)   285,943 条  (~25%)
#   4090 机器 0:        [571886 ~ 714857)   142,971 条  (~12.5%)
#   4090 机器 1:        [714857 ~ 857828)   142,971 条  (~12.5%)
#   4090 机器 2:        [857828 ~ 1000799)  142,971 条  (~12.5%)
#   4090 机器 3:        [1000799 ~ 1143771) 142,972 条  (~12.5%)
#
# ============================================================
# 用法 (在对应机器上分别执行):
#
#   H200 机器 0:
#     bash run_caption_all.sh h200 0
#
#   H200 机器 1:
#     bash run_caption_all.sh h200 1
#
#   4090 机器 0:
#     bash run_caption_all.sh 4090 0
#
#   4090 机器 1:
#     bash run_caption_all.sh 4090 1
#
#   4090 机器 2:
#     bash run_caption_all.sh 4090 2
#
#   4090 机器 3:
#     bash run_caption_all.sh 4090 3
#
# ============================================================
# 全部完成后，在任意一台执行合并:
#   python3 qwenvl_rewrite.py --merge-only \
#       --tags h200_0 h200_1 4090_m0 4090_m1 4090_m2 4090_m3
# ============================================================

set -e

CLUSTER="${1}"      # h200 或 4090
MACHINE_ID="${2}"   # 机器编号

if [ -z "${CLUSTER}" ] || [ -z "${MACHINE_ID}" ]; then
    echo "用法:"
    echo "  H200 机器 0:  bash run_caption_all.sh h200 0"
    echo "  H200 机器 1:  bash run_caption_all.sh h200 1"
    echo "  4090 机器 0:  bash run_caption_all.sh 4090 0"
    echo "  4090 机器 1:  bash run_caption_all.sh 4090 1"
    echo "  4090 机器 2:  bash run_caption_all.sh 4090 2"
    echo "  4090 机器 3:  bash run_caption_all.sh 4090 3"
    exit 1
fi

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
NUM_GPUS=8

# ============================================================
# 数据切分表 (总共 1,143,771 条)
# ============================================================
# H200 对半分:
#   h200_0: [     0,  285943)
#   h200_1: [285943,  571886)
# 4090 四等分:
#   m0:     [571886,  714857)
#   m1:     [714857,  857828)
#   m2:     [857828, 1000799)
#   m3:     [1000799, 1143771)

if [ "${CLUSTER}" = "h200" ]; then
    case "${MACHINE_ID}" in
        0) INPUT_FILE="${SCRIPT_DIR}/metadata_caption_h200_h200_0.json" ;;
        1) INPUT_FILE="${SCRIPT_DIR}/metadata_caption_h200_h200_1.json" ;;
        *) echo "错误: H200 机器编号需为 0 或 1"; exit 1 ;;
    esac
    TAG="h200_${MACHINE_ID}"
    # H200 用独立的 subset JSON，不需要 data-start/data-end
    EXTRA_ARGS="--input ${INPUT_FILE}"
elif [ "${CLUSTER}" = "4090" ]; then
    case "${MACHINE_ID}" in
        0) DATA_START=571886;  DATA_END=714857  ;;
        1) DATA_START=714857;  DATA_END=857828  ;;
        2) DATA_START=857828;  DATA_END=1000799 ;;
        3) DATA_START=1000799; DATA_END=1143771 ;;
        *) echo "错误: 4090 机器编号需为 0-3"; exit 1 ;;
    esac
    TAG="4090_m${MACHINE_ID}"
    EXTRA_ARGS="--data-start ${DATA_START} --data-end ${DATA_END}"
else
    echo "错误: 第一个参数需为 'h200' 或 '4090'"
    exit 1
fi

echo "============================================"
echo "Qwen3-VL-8B Caption 批量推理"
echo "  集群:      ${CLUSTER}"
echo "  机器编号:  ${MACHINE_ID}"
echo "  标签:      ${TAG}"
if [ "${CLUSTER}" = "h200" ]; then
    echo "  输入文件:  ${INPUT_FILE}"
else
    echo "  数据范围:  [${DATA_START:,} ~ ${DATA_END:,})"
fi
echo "  GPU 数:    ${NUM_GPUS}"
echo "  模型:      ${SCRIPT_DIR}/models/Qwen3-VL-8B-Instruct"
echo "  中断后重新运行相同命令即可断点续跑"
echo "============================================"
echo ""

cd "${SCRIPT_DIR}"

python3 qwenvl_rewrite.py \
    ${EXTRA_ARGS} \
    --tag "${TAG}" \
    --num-gpus "${NUM_GPUS}" \
    --model-path "${SCRIPT_DIR}/models/Qwen3-VL-8B-Instruct"

RET=$?
echo ""
if [ ${RET} -eq 0 ]; then
    echo "✅ ${TAG} 完成!"
    echo ""
    echo "全部完成后，在任意一台执行合并:"
    echo "  cd ${SCRIPT_DIR} && python3 qwenvl_rewrite.py --merge-only --tags h200 h200_0 h200_1 4090_m0 4090_m1 4090_m2 4090_m3"
else
    echo "❌ 异常退出 (exit code: ${RET})，重新运行即可断点续跑"
fi
