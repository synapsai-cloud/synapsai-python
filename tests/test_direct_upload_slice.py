import os

from synapsai.resources.model_artifacts import _FileSlice


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
