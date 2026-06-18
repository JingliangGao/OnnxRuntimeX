#!/bin/bash

# 配置文件路径
INPUT_FILE="onnx_path.txt"
DEST_ROOT="ort_hub"

# 复制模型到 model_hub，保持目录结构
copy_models() {
    if [ ! -f "$INPUT_FILE" ]; then
        echo "错误：找不到 $INPUT_FILE" >&2
        exit 1
    fi

    mkdir -p "$DEST_ROOT"

    while IFS= read -r line; do
        # 去除首尾空白字符
        line="$(echo "$line" | sed -e 's/^[[:space:]]*//' -e 's/[[:space:]]*$//')"
        [ -z "$line" ] && continue

        # 移除开头的 "./"（如果有）
        if [[ "$line" == ./* ]]; then
            rel_path="${line#./}"
        else
            rel_path="$line"
        fi

        src="$rel_path"
        dest="$DEST_ROOT/$rel_path"

        if [ ! -f "$src" ]; then
            echo "警告：源文件不存在，跳过 -> $src" >&2
            continue
        fi

        # 创建目标目录并复制文件
        mkdir -p "$(dirname "$dest")"
        cp "$src" "$dest"
        echo "已复制: $rel_path"
    done < "$INPUT_FILE"

    echo "复制完成。"
}

# 删除 model_hub 中的模型文件（保留空目录）
delete_models() {
    if [ ! -f "$INPUT_FILE" ]; then
        echo "错误：找不到 $INPUT_FILE" >&2
        exit 1
    fi

    if [ ! -d "$DEST_ROOT" ]; then
        echo "警告：$DEST_ROOT 不存在，无需删除。" >&2
        return
    fi

    while IFS= read -r line; do
        line="$(echo "$line" | sed -e 's/^[[:space:]]*//' -e 's/[[:space:]]*$//')"
        [ -z "$line" ] && continue

        if [[ "$line" == ./* ]]; then
            rel_path="${line#./}"
        else
            rel_path="$line"
        fi

        dest="$DEST_ROOT/$rel_path"

        if [ -f "$dest" ]; then
            rm -f "$dest"
            echo "已删除: $rel_path"
        else
            echo "警告：目标文件不存在，跳过 -> $dest" >&2
        fi
        # 注意：不删除目录，只删除文件
    done < "$INPUT_FILE"

    echo "删除完成（目录已保留）。"
}

# 显示使用帮助
usage() {
    echo "用法: $0 {copy|delete}"
    echo "  copy   - 将 find_result.txt 中列出的模型复制到 model_hub"
    echo "  delete - 删除 model_hub 中对应的模型文件（保留文件夹）"
    exit 1
}

# 主逻辑
case "$1" in
    copy)
        copy_models
        ;;
    delete)
        delete_models
        ;;
    *)
        usage
        ;;
esac
