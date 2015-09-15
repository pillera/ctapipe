from ..files import get_file_type


def test_get_file_type():

    # create  a dictionary of filename and expected type
    filenames = {'test.fits.gz': 'fits',
                 'test.fits': 'fits',
                 'test.fits.bz2': 'fits',
                 'test.fit': 'fits',
                 'test_file.eventio.gz': 'eventio',
                 'test_file.eventio': 'eventio'}

    for filename, filetype in filenames.items():
        assert get_file_type(filename) == filetype
