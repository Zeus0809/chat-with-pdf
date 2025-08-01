def test_temp_dir_fixture(temp_dir):
    """Test that the temp directory fixture works"""

    test_file = temp_dir / "hello.txt"
    test_file.write_text("Hello from test!")

    assert test_file.exists()
    assert test_file.read_text() == "Hello from test!"
