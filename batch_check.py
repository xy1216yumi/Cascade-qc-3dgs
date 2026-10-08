"""批量检查：对 sample 目录（或指定目录）下所有渲染图逐张执行检查，结果写入 JSONL。

用法：
    python batch_check.py                          # 默认检查 sample/，输出 results.jsonl
    python batch_check.py --input 图片目录 --output 结果.jsonl --limit 20

每次运行自动跳过输出文件中已存在的图片（断点续跑）。
"""
import argparse
import json
import os
import sys

import config
from checker import check_image

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

IMG_EXTS = {".png", ".jpg", ".jpeg"}


def load_done(results_path: str) -> set[str]:
    """已成功处理的图片集合（调用失败的记录不标记为完成，重跑可再试）。"""
    done: set[str] = set()
    if os.path.exists(results_path):
        with open(results_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    rec = json.loads(line)
                    if rec.get("instances"):
                        done.add(rec["image"])
                except (json.JSONDecodeError, KeyError):
                    continue
    return done


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", default=config.DEFAULT_INPUT_DIR)
    ap.add_argument("--output", default=config.DEFAULT_RESULTS)
    ap.add_argument("--limit", type=int, default=0, help="最多处理多少张（0=全部）")
    ap.add_argument("--no-reflect", action="store_true", help="关闭反思闭环（对比用）")
    args = ap.parse_args()

    err = config.check_config()
    if err:
        print(err)
        sys.exit(1)

    if not os.path.isdir(args.input):
        print(f"输入目录不存在: {args.input}")
        sys.exit(1)

    images = sorted(
        os.path.join(args.input, n)
        for n in os.listdir(args.input)
        if os.path.splitext(n)[1].lower() in IMG_EXTS
    )
    if not images:
        print(f"目录 {args.input} 下没有图片（支持 png/jpg/jpeg）")
        sys.exit(1)

    done = load_done(args.output)
    todo = [p for p in images if p not in done]
    print(f"共 {len(images)} 张图，已完成 {len(images) - len(todo)} 张，本次处理 {len(todo)} 张")
    if args.limit > 0:
        todo = todo[: args.limit]
        print(f"（limit={args.limit}）")

    ok = 0
    fail = 0
    with open(args.output, "a", encoding="utf-8") as out:
        for i, img in enumerate(todo, 1):
            print(f"[{i}/{len(todo)}] {os.path.basename(img)} ...")
            try:
                result = check_image(img, reflect=not args.no_reflect)
                out.write(json.dumps(result, ensure_ascii=False) + "\n")
                out.flush()
                ok += 1
            except Exception as e:
                fail += 1
                print(f"  [失败] {type(e).__name__}: {e}")
                # 失败也记录一行，便于排查（instances 为空）
                out.write(
                    json.dumps(
                        {"image": img, "instances": [], "error": f"{type(e).__name__}: {e}"},
                        ensure_ascii=False,
                    )
                    + "\n"
                )
                out.flush()

    print(f"\n完成：成功 {ok}，失败 {fail}。结果已追加到 {args.output}")


if __name__ == "__main__":
    main()
