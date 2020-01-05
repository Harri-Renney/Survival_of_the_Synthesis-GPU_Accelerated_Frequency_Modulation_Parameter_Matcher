#ifndef FFTW_WRAPPER_HPP
#define FFTW_WRAPPER_HPP

#include <cstdint>

#include "fftw3.h"

class FFTW_Wrapper
{
private:
	double *window;				//Buffer for the window.
	double *windowed_audio;		//Buffer for the windowed audio.

	//Window factor is sum of the window samples divided by the FFT size//
	float window_factor;
	float one_over_window_factor;

	uint32_t size_log2;
	uint32_t size;
	uint32_t half_size;
	float one_over_size;

	fftw_complex* outputFFT;
public:
	FFTW_Wrapper()
	{

	}
	void init()
	{

	}
	void calculate()
	{

	}
};

#endif