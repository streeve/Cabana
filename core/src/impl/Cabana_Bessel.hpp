/****************************************************************************
 * Copyright (c) 2023 by the Cabana authors                                 *
 * All rights reserved.                                                     *
 *                                                                          *
 * This file is part of the Cabana library. Cabana is distributed under a   *
 * BSD 3-clause license. For the licensing terms see the LICENSE file in    *
 * the top-level directory.                                                 *
 *                                                                          *
 * SPDX-License-Identifier: BSD-3-Clause                                    *
 ****************************************************************************/

/*!
  \file impl/Cabana_GaussianMixtureModel.hpp
  \brief Creation of a Gaussian Mixture Model
*/
#ifndef CABANA_BESSEL_IMPL_HPP
#define CABANA_BESSEL_IMPL_HPP

#include <cmath>

namespace Cabana {

namespace Impl {

class Bessel {

private:
	static double KOKKOS_INLINE_FUNCTION p13(const double x) {
		const double P0  = 8.3333333333333333311567967e-02;
		const double P1  = 6.9444444444444450632369337e-03;
		const double P2  = 3.4722222222221933634809047e-04;
		const double P3  = 1.1574074074079326676719210e-05;
		const double P4  = 2.7557319223490964726712181e-07;
		const double P5  = 4.9209498642000488498034902e-09;
		const double P6  = 6.8346524852208360284643288e-11;
		const double P7  = 7.5940608652484019265663823e-13;
		const double P8  = 6.9036483611746228414130722e-15;
		const double P9  = 5.2305536429160706017626195e-17;
		const double P10 = 3.3486060280590464196327437e-19;
		const double P11 = 1.8645262719811753663834319e-21;
		const double P12 = 7.9611250107842314599760659e-24;
		const double P13 = 5.3251032089995165438568695e-26;
		return P0 + x*(P1 + x*(P2 + x*(P3 + x*(P4 + x*(P5 + x*(P6 + x*(P7 + x*(P8 + x*(P9 + x*(P10 + x*(P11 + x*(P12 + x*P13))))))))))));
	}

	static double KOKKOS_INLINE_FUNCTION p16(const double x) {
		const double P0  = 1.0000000000000000000000801e+00;
		const double P1  = 2.4999999999999999999629693e-01;
		const double P2  = 2.7777777777777777805664954e-02;
		const double P3  = 1.7361111111111110294015271e-03;
		const double P4  = 6.9444444444444568581891535e-05;
		const double P5  = 1.9290123456788994104574754e-06;
		const double P6  = 3.9367598891475388547279760e-08;
		const double P7  = 6.1511873265092916275099070e-10;
		const double P8  = 7.5940584360755226536109511e-12;
		const double P9  = 7.5940582595094190098755663e-14;
		const double P10 = 6.2760839879536225394314453e-16;
		const double P11 = 4.3583591008893599099577755e-18;
		const double P12 = 2.5791926805873898803749321e-20;
		const double P13 = 1.3141332422663039834197910e-22;
		const double P14 = 5.9203280572170548134753422e-25;
		const double P15 = 2.0732014503197852176921968e-27;
		const double P16 = 1.1497640034400735733456400e-29;

		return P0 + x*(P1 + x*(P2 + x*(P3 + x*(P4 + x*(P5 + x*(P6 + x*(P7 + x*(P8 + x*(P9 + x*(P10 + x*(P11 + x*(P12 + x*(P13 + x*(P14 + x*(P15 + P16*x)))))))))))))));
	}

	static double KOKKOS_INLINE_FUNCTION p22a(const double x) {
		const double P0  = 3.9894228040143265335649948e-01;
		const double P1  = 4.9867785050353992900698488e-02;
		const double P2  = 2.8050628884163787533196746e-02;
		const double P3  = 2.9219501690198775910219311e-02;
		const double P4  = 4.4718622769244715693031735e-02;
		const double P5  = 9.4085204199017869159183831e-02;
		const double P6  = -1.0699095472110916094973951e-01;
		const double P7  = 2.2725199603010833194037016e+01;
		const double P8  = -1.0026890180180668595066918e+03;
		const double P9  = 3.1275740782277570164423916e+04;
		const double P10 = -5.9355022509673600842060002e+05;
		const double P11 = 2.6092888649549172879282592e+06;
		const double P12 = 2.3518420447411254516178388e+08;
		const double P13 = -8.9270060370015930749184222e+09;
		const double P14 = 1.8592340458074104721496236e+11;
		const double P15 = -2.6632742974569782078420204e+12;
		const double P16 = 2.7752144774934763122129261e+13;
		const double P17 = -2.1323049786724612220362154e+14;
		const double P18 = 1.1989242681178569338129044e+15;
		const double P19 = -4.8049082153027457378879746e+15;
		const double P20 = 1.3012646806421079076251950e+16;
		const double P21 = -2.1363029690365351606041265e+16;
		const double P22 = 1.6069467093441596329340754e+16;

		return P0 + x*(P1 + x*(P2 + x*(P3 + x*(P4 + x*(P5 + x*(P6 + x*(P7 + x*(P8 + x*(P9 + x*(P10 + x*(P11 + x*(P12 + x*(P13 + x*(P14 + x*(P15 + x*(P16 + x*(P17 + x*(P18 + x*(P19 + x*(P20 + x*(P21 + x*P22)))))))))))))))))))));
	}

	static double KOKKOS_INLINE_FUNCTION p22b(const double x) {
		const double P0  = 3.9894228040143270388374079e-01;
		const double P1  = -1.4960335515072058522575487e-01;
		const double P2  = -4.6751048269476797374239762e-02;
		const double P3  = -4.0907267094886972971863462e-02;
		const double P4  = -5.7501487840859800117669379e-02;
		const double P5  = -1.1428156617865937773864845e-01;
		const double P6  = 6.7988447242260666801129937e-02;
		const double P7  = -2.2694203870019250176636896e+01;
		const double P8  = 9.7548286270114208672947525e+02;
		const double P9  = -2.9286459257939415083570152e+04;
		const double P10 = 4.9934855620495985742805154e+05;
		const double P11 = 5.7682364160056137069002930e+05;
		const double P12 = -3.1576840778898356890175020e+08;
		const double P13 = 1.0484906321376589515223174e+10;
		const double P14 = -2.0918193917759394367113655e+11;
		const double P15 = 2.9320804098307168426392082e+12;
		const double P16 = -3.0147278411132255281401004e+13;
		const double P17 = 2.2950466603697814797615042e+14;
		const double P18 = -1.2816007548999035598180100e+15;
		const double P19 = 5.1086996139908353110844064e+15;
		const double P20 = -1.3774917783425787550429723e+16;
		const double P21 = 2.2531580094188348024267027e+16;
		const double P22 = -1.6895178303473738478791245e+16;

		return P0 + x*(P1 + x*(P2 + x*(P3 + x*(P4 + x*(P5 + x*(P6 + x*(P7 + x*(P8 + x*(P9 + x*(P10 + x*(P11 + x*(P12 + x*(P13 + x*(P14 + x*(P15 + x*(P16 + x*(P17 + x*(P18 + x*(P19 + x*(P20 + x*(P21 + x*P22)))))))))))))))))))));
	}


public:
	static double KOKKOS_INLINE_FUNCTION i0_approx(const double x) {
		// Ideally we would just call std::cyl_bessel_i(0,x), but that is only
		// available on the host, not on GPUs so we are going to use the series
		// expansion from
		// https://www.advanpix.com/2015/11/11/rational-approximations-for-the-modified-bessel-function-of-the-first-kind-i0-computations-double-precision/
		// that also ended up in boost
		const double x_half = 0.5 * x;
		if(x <= 7.75) {
			return 1. + x_half*x_half * p16(x_half*x_half);
		} else {
			return exp(x_half) * p22a(1./x) / sqrt(x) * exp(x_half);
		}
	}

	static double KOKKOS_INLINE_FUNCTION i1_approx(const double x) {
		// Ideally we would just call std::cyl_bessel_i(1,x), but that is only
		// available on the host, not on GPUs so we are going to use the series
		// expansion from
		// https://www.advanpix.com/2015/11/12/rational-approximations-for-the-modified-bessel-function-of-the-first-kind-i1-for-computations-with-double-precision/
		// that also ended up in boost
		if(x <= 7.75) {
			const double x_half = 0.5*x;
			return x_half*(1. + x_half*x_half*(0.5 + x_half*x_half*p13(x_half*x_half)));
		} else {
			return exp(x) * p22b(1./x) / sqrt(x);
		}
	}
};

} // end namespace Impl

} // end namespace Cabana

#endif // end CABANA_BESSEL_IMPL_HPP
