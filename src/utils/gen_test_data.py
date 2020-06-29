from ..spikeloader import SpikeLoader, Path_str


def gen_test_data(path: Path_str = 'data/superstim32.npz', save_path: Path_str = 'tests/data/test_data.hdf5',
                  t: int = None, n_neu: int = 1000):
    loader = SpikeLoader.from_npz(path)

    loader.pos = loader.pos.iloc[:n_neu]
    loader.istim = loader.istim.loc[:t]  # Index is t.
    loader.spks = loader.spks[:t, :n_neu]
    loader.save(save_path, overwrite=True, )


if __name__ == '__main__':
    gen_test_data(save_path='tests/data/raw.hdf5')
    test = SpikeLoader.from_hdf5('tests/data/raw.hdf5')
    test.save_processed('tests/data/processed.hdf5', overwrite=True)
