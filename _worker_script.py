import os
import multiprocessing as mp
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

def gpu_worker(worker_id, gpu_id, image_paths, output_root, output_size=112):
    # Pin this process to a specific GPU index
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)

    from insightface.app import FaceAnalysis
    from insightface.utils import face_align

    print(f"[Worker {worker_id} | GPU {gpu_id}] Processing {len(image_paths)} images")

    try:
        app = FaceAnalysis(
            name="buffalo_l",
            allowed_modules=["detection"],
            providers=["CUDAExecutionProvider", "CPUExecutionProvider"],
        )
        app.prepare(ctx_id=0, det_size=(640, 640))
    except Exception as e:
        print(f"[Worker {worker_id}] Init error: {e}")
        return 0, 0, len(image_paths)

    processed = 0
    missed = 0
    errors = 0

    for img_path_str, rel_path in image_paths:
        try:
            input_path = Path(img_path_str)
            out_path = Path(output_root) / rel_path
            out_path.parent.mkdir(parents=True, exist_ok=True)

            img = Image.open(input_path).convert("RGB")
            arr = np.array(img)
            bgr = cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)

            faces = app.get(bgr)

            if faces:
                largest = max(
                    faces,
                    key=lambda f: (f.bbox[2]-f.bbox[0]) * (f.bbox[3]-f.bbox[1]),
                )
                if getattr(largest, "kps", None) is not None:
                    crop = face_align.norm_crop(bgr, largest.kps, image_size=output_size)
                    final_img = Image.fromarray(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB))
                else:
                    b = largest.bbox.astype(int)
                    h, w, _ = arr.shape
                    x1, y1 = max(0, b[0]), max(0, b[1])
                    x2, y2 = min(w, b[2]), min(h, b[3])
                    crop = arr[y1:y2, x1:x2]
                    final_img = Image.fromarray(crop).resize((output_size, output_size))

                final_img.convert("L").save(out_path)
                processed += 1
            else:
                missed += 1
                Image.fromarray(arr).convert("L").resize(
                    (output_size, output_size)
                ).save(out_path)

        except Exception:
            errors += 1

        if processed > 0 and processed % 500 == 0:
            print(f"[Worker {worker_id}] {processed}/{len(image_paths)} processed")

    return processed, missed, errors


def main():
    input_dir = Path(os.environ.get("INPUT_DIR"))
    output_dir = Path(os.environ.get("OUTPUT_DIR"))

    print(f"[Master] Scanning {input_dir} ...")

    tasks = []
    for root, _, files in os.walk(input_dir):
        for f in files:
            if f.lower().endswith((".png", ".jpg", ".jpeg")):
                full = Path(root) / f
                rel = full.relative_to(input_dir)
                tasks.append((str(full), str(rel)))

    total = len(tasks)
    print(f"[Master] Found {total} images")
    if total == 0:
        return

    visible = os.environ.get("CUDA_VISIBLE_DEVICES", "")
    gpus = [x for x in visible.split(",") if x.strip()]
    num_gpus = len(gpus) or 1

    print(f"[Master] Using {num_gpus} GPU workers")

    chunk = int(np.ceil(total / num_gpus))
    chunks = [tasks[i:i+chunk] for i in range(0, total, chunk)]

    with mp.Pool(processes=num_gpus) as pool:
        results = [
            pool.apply_async(gpu_worker, (i, i % num_gpus, c, str(output_dir)))
            for i, c in enumerate(chunks)
        ]

        processed = missed = errors = 0
        for r in results:
            p, m, e = r.get()
            processed += p
            missed += m
            errors += e

    print(f"[SUMMARY] processed={processed} missed={missed} errors={errors}")


if __name__ == "__main__":
    main()
