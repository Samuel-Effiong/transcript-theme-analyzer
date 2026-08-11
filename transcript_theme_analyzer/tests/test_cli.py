import os

from transcript_theme_analyzer.cli import _discover_transcript_paths


def test_discover_transcript_paths_recursive_and_filtering(tmp_path):
    # Setup directory structure with subfolders, hidden files, and temp lock files
    sub1 = tmp_path / "folder_a"
    sub2 = tmp_path / "folder_b" / "subfolder"
    sub1.mkdir(parents=True)
    sub2.mkdir(parents=True)

    # Valid transcripts
    (tmp_path / "root.txt").write_text("root txt")
    (sub1 / "file1.docx").write_text("docx 1")
    (sub2 / "file2.DOCX").write_text("docx 2 uppercase")
    (sub2 / "file3.Txt").write_text("txt 3 mixedcase")

    # Files that should be ignored
    (tmp_path / ".hidden.txt").write_text("hidden")
    (sub1 / "._file1.docx").write_text("mac OS metadata")
    (sub1 / "~$file1.docx").write_text("word lock file")
    (sub2 / "notes.pdf").write_text("unsupported pdf")

    discovered = _discover_transcript_paths(str(tmp_path), glob_pattern=None)

    relative_discovered = [os.path.relpath(p, str(tmp_path)) for p in discovered]

    assert relative_discovered == sorted([
        "folder_b/subfolder/file2.DOCX",
        "folder_b/subfolder/file3.Txt",
        "folder_a/file1.docx",
        "root.txt",
    ])
