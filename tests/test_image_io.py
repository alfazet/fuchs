import numpy as np
import pytest
from PIL import Image

from image_io import BITS_PER_SIDE, BLOCK_SIZE, GRID_SIZE, read_bits, write_bits


def random_bits(seed=2137):
    rng = np.random.default_rng(seed)
    bits = rng.integers(0, 2, size=(BITS_PER_SIDE, BITS_PER_SIDE), dtype=np.uint8)
    bits[0, 0] = 1  # ensure a black top-left corner

    return bits


def bits_to_pixels(bits):
    block = np.ones((BLOCK_SIZE, BLOCK_SIZE), dtype=np.uint8)
    return (1 - np.kron(bits.astype(np.uint8), block)) * 255


def embed_pixels(pixels, image_w, image_h, x, y, fill=255):
    image = np.full((image_h, image_w), fill, dtype=np.uint8)
    h, w = pixels.shape
    image[y : y + h, x : x + w] = pixels

    return image


def save_pixels(pixels, path):
    Image.fromarray(pixels, mode="L").save(path)


def all_black_grid():
    return np.zeros((GRID_SIZE, GRID_SIZE), dtype=np.uint8)


class TestConstants:
    def test_dimensions(self):
        assert GRID_SIZE % BLOCK_SIZE == 0


class TestReadBits:
    def test_shape_and_datatype(self, tmp_path):
        path = tmp_path / "in.png"
        save_pixels(bits_to_pixels(random_bits()), path)
        result = read_bits(path)
        assert result.shape == (BITS_PER_SIDE, BITS_PER_SIDE)
        assert result.dtype == np.uint8
        assert set(np.unique(result).tolist()) <= {0, 1}

    def test_all_black_grid(self, tmp_path):
        path = tmp_path / "in.png"
        save_pixels(all_black_grid(), path)
        assert (read_bits(path) == 1).all()

    @pytest.mark.parametrize("seed", range(3))
    def test_round_trip(self, tmp_path, seed):
        bits = random_bits(seed)
        path = tmp_path / "out.png"
        write_bits(bits, path)
        assert np.array_equal(read_bits(path), bits)

    def test_transparent_pixels_count_as_white(self, tmp_path):
        bits = random_bits()
        pixels = bits_to_pixels(bits)
        rgba = np.zeros((GRID_SIZE, GRID_SIZE, 4), dtype=np.uint8)
        rgba[..., :3] = pixels[..., None]
        rgba[..., 3] = 255
        white = pixels == 255
        rgba[white, :3] = 0  # make all white pixels black and fully transparent
        rgba[white, 3] = 0
        path = tmp_path / "in.png"
        Image.fromarray(rgba, mode="RGBA").save(path)
        assert np.array_equal(read_bits(path), bits)


class TestWriteBits:
    def test_output_image_properties(self, tmp_path):
        path = tmp_path / "out.png"
        write_bits(random_bits(), path)
        with Image.open(path) as img:
            assert img.format == "PNG"
            assert img.size == (GRID_SIZE, GRID_SIZE)

    def test_only_black_and_white_pixels(self, tmp_path):
        path = tmp_path / "out.png"
        write_bits(random_bits(), path)
        with Image.open(path) as img:
            values = np.unique(np.asarray(img.convert("L")))
        assert set(values.tolist()) <= {0, 255}

    def test_single_bit_maps_to_correct_block(self, tmp_path):
        path = tmp_path / "out.png"
        bits = np.zeros((BITS_PER_SIDE, BITS_PER_SIDE), dtype=np.uint8)
        row, col = 12, 25
        bits[row, col] = 1
        write_bits(bits, path)
        with Image.open(path) as img:
            pixels = np.asarray(img.convert("L"))
        black = pixels == 0
        assert black.sum() == BLOCK_SIZE * BLOCK_SIZE
        assert black[
            row * BLOCK_SIZE : (row + 1) * BLOCK_SIZE,
            col * BLOCK_SIZE : (col + 1) * BLOCK_SIZE,
        ].all()

    @pytest.mark.parametrize(
        "shape",
        [
            (BITS_PER_SIDE,),
            (BITS_PER_SIDE - 1, BITS_PER_SIDE),
            (BITS_PER_SIDE // 2, BITS_PER_SIDE // 2),
            (BITS_PER_SIDE, BITS_PER_SIDE + 1),
            (BITS_PER_SIDE, BITS_PER_SIDE, 1),
        ],
    )
    def test_wrong_shape_raises(self, tmp_path, shape):
        with pytest.raises(ValueError):
            write_bits(np.zeros(shape, dtype=np.uint8), tmp_path / "out.png")

    @pytest.mark.parametrize("bad_value", [21, 37, 1225])
    def test_non_binary_values_raise(self, tmp_path, bad_value):
        bits = np.zeros((BITS_PER_SIDE, BITS_PER_SIDE), dtype=np.int64)
        bits[5, 5] = bad_value
        with pytest.raises(ValueError):
            write_bits(bits, tmp_path / "out.png")

    def test_invalid_input_does_not_create_file(self, tmp_path):
        path = tmp_path / "out.png"
        with pytest.raises(ValueError):
            write_bits(np.zeros((21, 37)), path)
        assert not path.exists()


class TestThresholdAndAveraging:
    def _grid_with_block(self, block_pixels):
        pixels = np.full((GRID_SIZE, GRID_SIZE), 255, dtype=np.uint8)
        pixels[0:4, 0:4] = 0
        pixels[0:4, 4:8] = block_pixels

        return pixels

    def _read_block_bit(self, tmp_path, block_pixels):
        path = tmp_path / "in.png"
        save_pixels(self._grid_with_block(block_pixels), path)

        return read_bits(path)[0, 1]

    def test_mean_above_half_is_zero(self, tmp_path):
        block = np.full((4, 4), 0, dtype=np.uint8)
        block.flat[:9] = 255  # 9/16 > 0.5
        assert self._read_block_bit(tmp_path, block) == 0

    def test_mean_less_than_or_equal_half_is_one(self, tmp_path):
        block = np.full((4, 4), 0, dtype=np.uint8)
        block.flat[:7] = 255  # 7/16 <= 0.5
        assert self._read_block_bit(tmp_path, block) == 1


class TestGridLocation:
    @pytest.mark.parametrize("x,y", [(0, 0), (1, 0), (0, 1), (12, 25), (21, 37)])
    def test_offset_grid_is_found(self, tmp_path, x, y):
        bits = random_bits(2)
        pixels = embed_pixels(bits_to_pixels(bits), GRID_SIZE + x, GRID_SIZE + y, x, y)
        path = tmp_path / "in.png"
        save_pixels(pixels, path)
        assert np.array_equal(read_bits(path), bits)

    def test_stray_dark_pixel_left_of_grid_shifts_the_corner(self, tmp_path):
        bits = random_bits(3)
        pixels = embed_pixels(
            bits_to_pixels(bits), GRID_SIZE + 37, GRID_SIZE + 37, 21, 21
        )
        pixels[12, 25] = 0
        path = tmp_path / "in.png"
        save_pixels(pixels, path)
        shifted = read_bits(path)  # decoding succeeds but with a different result
        assert shifted.shape == (BITS_PER_SIDE, BITS_PER_SIDE)
        assert not np.array_equal(shifted, bits)

    def test_extra_area_is_ignored(self, tmp_path):
        bits = random_bits(7)
        pixels = np.full((GRID_SIZE + 12, GRID_SIZE + 25), 0, dtype=np.uint8)
        pixels[:GRID_SIZE, :GRID_SIZE] = bits_to_pixels(bits)
        path = tmp_path / "in.png"
        save_pixels(pixels, path)
        result = read_bits(path)
        assert result.shape == (BITS_PER_SIDE, BITS_PER_SIDE)
        assert np.array_equal(result, bits)


class TestErrors:
    def test_all_white_image_raises(self, tmp_path):
        path = tmp_path / "in.png"
        save_pixels(np.full((GRID_SIZE, GRID_SIZE), 255, dtype=np.uint8), path)
        with pytest.raises(ValueError):
            read_bits(path)

    def test_image_smaller_than_grid_raises(self, tmp_path):
        path = tmp_path / "in.png"
        save_pixels(np.zeros((21, 37), dtype=np.uint8), path)
        with pytest.raises(ValueError):
            read_bits(path)

    def test_too_narrow_raises(self, tmp_path):
        path = tmp_path / "in.png"
        save_pixels(np.zeros((GRID_SIZE, GRID_SIZE - 12), dtype=np.uint8), path)
        with pytest.raises(ValueError):
            read_bits(path)

    def test_too_short_raises(self, tmp_path):
        path = tmp_path / "in.png"
        save_pixels(np.zeros((GRID_SIZE - 25, GRID_SIZE), dtype=np.uint8), path)
        with pytest.raises(ValueError):
            read_bits(path)

    def test_offset_leaves_too_little_room_raises(self, tmp_path):
        pixels = np.full((GRID_SIZE, GRID_SIZE), 255, dtype=np.uint8)
        pixels[12:, 25:] = 0
        path = tmp_path / "in.png"
        save_pixels(pixels, path)
        with pytest.raises(ValueError):
            read_bits(path)

    def test_missing_file_raises(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            read_bits(tmp_path / "does_not_exist.png")
