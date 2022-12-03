import twoD_functions as myfunc
from twoD_settings import*

from timeit import default_timer as timer
import warnings

warnings.simplefilter("ignore", np.ComplexWarning) # warning turned off 


if __name__ == "__main__":

    Iin, g, Voc = myfunc.loadFEKO("crossblock")
    ground = myfunc.get_ground_truth("crossblock")
    
    Gmat = myfunc.g_to_Gmatrix(g,Iin)
    # Gmat = myfunc.tsvd(Gmat)
    Voc_holed, mask = myfunc.undersampling(Voc, 0.25)

    start = timer()
    print("===============without compressed sensing==================")
    Tvec, Tmat = myfunc.voc_to_Tvec(Voc_holed, fratio)
    angular_spectrum = myfunc.least_square_argument(Gmat, Tvec)
    myfunc.twok_filter(angular_spectrum)
    epsr = myfunc.angspectrum_to_image(angular_spectrum)
    myfunc.print_stats(epsr, ground)
    print("elapsed time = ", timer()-start)

    second = timer()
    Voc_filled = myfunc.compressed_sensing(Voc_holed, mask)
    print("===============with compressed sensing=====================")
    Tvec2, Tmat2 = myfunc.voc_to_Tvec(Voc_filled, fratio)
    angular_spectrum2 = myfunc.least_square_argument(Gmat, Tvec2)
    myfunc.twok_filter(angular_spectrum2)
    epsr2 = myfunc.angspectrum_to_image(angular_spectrum2)
    myfunc.print_stats(epsr2, ground)
    print("elapsed time = ", timer()-second)
    
    # myfunc.plot_ground_truth(ground)
    myfunc.plot_angular_spectrum(angular_spectrum)
    myfunc.plot_epsr(epsr)
    # myfunc.plot_Voc_spectrum(Tmat)
    myfunc.plot_angular_spectrum(angular_spectrum2)
    myfunc.plot_epsr(epsr2)
    # myfunc.plot_Voc_spectrum(Tmat2)
    myfunc.show_plot()


