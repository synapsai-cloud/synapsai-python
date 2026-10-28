import os
import threading
import time
from pathlib import Path

from synapsai.resources import model_artifacts
from synapsai.resources.model_artifacts import _FileSlice, _upload_presigned_parts


def test_file_slice_reads_only_its_window(tmp_path):
    path = tmp_path / "weights.bin"
    path.write_bytes(b"abcdefghij")
    with path.open("rb") as handle:
        handle.seek(2)
        window = _FileSlice(handle, 4)
        assert window.seek(0, os.SEEK_END) == 4
        window.seek(0)
        assert window.read() == b"cdef"
        assert handle.read(1) == b"g"


def test_presigned_parts_overlap_and_complete_in_part_order(monkeypatch):
    current = 0
    peak = 0
    lock = threading.Lock()

    def fake_put(url, file_path, offset, length, timeout):
        nonlocal current, peak
        with lock:
            current += 1
            peak = max(peak, current)
        try:
            time.sleep(0.05)
            return f"etag-{url}"
        finally:
            with lock:
                current -= 1

    monkeypatch.setattr(model_artifacts, "_put_presigned", fake_put)
    plan = {
        "mode": "multipart",
        "parts": [
            {"url": "b", "offset": 64, "size_bytes": 64, "part_number": 2},
            {"url": "a", "offset": 0, "size_bytes": 64, "part_number": 1},
            {"url": "c", "offset": 128, "size_bytes": 10, "part_number": 3},
        ],
    }
    finished = _upload_presigned_parts(plan, Path("unused"), 1)
    assert [item["part_number"] for item in finished] == [1, 2, 3]
    assert [item["etag"] for item in finished] == ["etag-a", "etag-b", "etag-c"]
    assert peak > 1
