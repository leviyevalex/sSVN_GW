from math import pi

import jax
import jax.numpy as jnp
from .IMRPhenomD_utils import (
    get_coeffs,
    get_delta0,
    get_delta1,
    get_delta2,
    get_delta3,
    get_delta4,
    get_transition_frequencies,
)

from ..constants import EulerGamma, gt, m_per_Mpc, C
from ..typing import Array
from models.ripple import Mc_eta_to_ms


def get_inspiral_phase(fM_s: Array, theta: Array, coeffs: Array) -> Array:
    # First lets calculate some of the vairables that will be used below
    # Mass variables
    m1, m2, chi1, chi2 = theta
    m1_s = m1 * gt
    m2_s = m2 * gt
    M_s = m1_s + m2_s
    eta = m1_s * m2_s / (M_s ** 2.0)

    # First lets construct the phase in the inspiral (region I)
    m1M = m1_s / M_s
    m2M = m2_s / M_s

    phi0 = 1.0
    phi1 = 0.0
    phi2 = 5.0 * (743.0 / 84.0 + 11.0 * eta) / 9.0
    phi3 = -16.0 * pi + (
        m1M * (25.0 + 38.0 / 3.0 * m1M) * chi1 + m2M * (25.0 + 38.0 / 3.0 * m2M) * chi2
    )
    phi4 = 5.0 * (3058.673 / 7.056 + 5429.0 / 7.0 * eta + 617.0 * eta * eta) / 72.0
    phi4 += (
        (247.0 / 4.8 * eta) * chi1 * chi2
        + (-721.0 / 4.8 * eta) * chi1 * chi2
        + ((-720.0 / 9.6 * m1M * m1M) + (1.0 / 9.6 * m1M * m1M)) * chi1 * chi1
        + ((-720.0 / 9.6 * m2M * m2M) + (1.0 / 9.6 * m2M * m2M)) * chi2 * chi2
        + ((240.0 / 9.6 * m1M * m1M) + (-7.0 / 9.6 * m1M * m1M)) * chi1 * chi1
        + ((240.0 / 9.6 * m2M * m2M) + (-7.0 / 9.6 * m2M * m2M)) * chi2 * chi2
    )
    phi5 = 5.0 / 9.0 * (7729.0 / 84.0 - 13.0 * eta) * pi
    phi5 += (
        -m1M
        * (
            1391.5 / 8.4
            - m1M * (1.0 - m1M) * 10.0 / 3.0
            + m1M * (1276.0 / 8.1 + m1M * (1.0 - m1M) * 170.0 / 9.0)
        )
    ) * chi1 + (
        -m2M
        * (
            1391.5 / 8.4
            - m2M * (1.0 - m2M) * 10.0 / 3.0
            + m2M * (1276.0 / 8.1 + m2M * (1.0 - m2M) * 170.0 / 9.0)
        )
    ) * chi2
    phi5_log = (5.0 / 3.0) * (7729.0 / 84.0 - 13.0 * eta) * pi
    phi5_log += 3.0 * (
        (
            -m1M
            * (
                1391.5 / 8.4
                - m1M * (1.0 - m1M) * 10.0 / 3.0
                + m1M * (1276.0 / 8.1 + m1M * (1.0 - m1M) * 170.0 / 9.0)
            )
        )
        * chi1
        + (
            -m2M
            * (
                1391.5 / 8.4
                - m2M * (1.0 - m2M) * 10.0 / 3.0
                + m2M * (1276.0 / 8.1 + m2M * (1.0 - m2M) * 170.0 / 9.0)
            )
        )
        * chi2
    )

    phi6 = (
        (
            11583.231236531 / 4.694215680
            - 640.0 / 3.0 * pi * pi
            - 6848.0 / 21.0 * EulerGamma
        )
        + eta * (-15737.765635 / 3.048192 + 2255.0 / 12.0 * pi * pi)
        + eta * eta * 76055.0 / 1728.0
        - eta * eta * eta * 127825.0 / 1296.0
        + (-6848.0 / 21.0) * jnp.log(4.0)
    )
    phi6 += (pi * m1M * (1490.0 / 3.0 + m1M * 260.0)) * chi1 + (
        pi * m2M * (1490.0 / 3.0 + m2M * 260.0)
    ) * chi2
    phi6_log = -6848.0 / 21.0

    phi7 = pi * (
        770.96675 / 2.54016 + 378.515 / 1.512 * eta - 740.45 / 7.56 * eta * eta
    )
    phi7 += (
        m1M
        * (
            -17097.8035 / 4.8384
            + eta * 28764.25 / 6.72
            + eta * eta * 47.35 / 1.44
            + m1M
            * (
                -7189.233785 / 1.524096
                + eta * 458.555 / 3.024
                - eta * eta * 534.5 / 7.2
            )
        )
    ) * chi1 + (
        m2M
        * (
            -17097.8035 / 4.8384
            + eta * 28764.25 / 6.72
            + eta * eta * 47.35 / 1.44
            + m2M
            * (
                -7189.233785 / 1.524096
                + eta * 458.555 / 3.024
                - eta * eta * 534.5 / 7.2
            )
        )
    ) * chi2

    # Add frequency dependence here
    v = (pi * fM_s) ** (1.0 / 3.0)

    phi_TF2 = (
        (phi5 - pi / 4)
        + phi0 * ((pi * fM_s) ** -(5.0 / 3.0))
        + phi1 * ((pi * fM_s) ** -(4.0 / 3.0))
        + phi2 * ((pi * fM_s) ** -1.0)
        + phi3 * ((pi * fM_s) ** -(2.0 / 3.0))
        + phi4 * ((pi * fM_s) ** -(1.0 / 3.0))
        + phi5_log * jnp.log(v)
        + phi6_log * jnp.log(v) * ((pi * fM_s) ** (1.0 / 3.0))
        + phi6 * ((pi * fM_s) ** (1.0 / 3.0))
        + phi7 * ((pi * fM_s) ** (2.0 / 3.0))
    ) * (3.0 / (128.0 * eta))

    phi_Ins = (
        phi_TF2
        + (
            coeffs[7] * fM_s
            + (3.0 / 4.0) * coeffs[8] * (fM_s ** (4.0 / 3.0))
            + (3.0 / 5.0) * coeffs[9] * (fM_s ** (5.0 / 3.0))
            + (1.0 / 2.0) * coeffs[10] * (fM_s ** 2.0)
        )
        / eta
    )
    return phi_Ins


# def get_inspiral_phase(fM_s: Array, theta: Array, coeffs: Array) -> Array:
#     # First lets calculate some of the vairables that will be used below
#     # Mass variables
#     m1, m2, chi1, chi2 = theta
#     m1_s = m1 * gt
#     m2_s = m2 * gt
#     M_s = m1_s + m2_s
#     eta = m1_s * m2_s / (M_s ** 2.0)
#     delta = (m1_s - m2_s) / M_s

#     # Spin variable
#     chi_s = (chi1 + chi2) / 2.0
#     chi_a = (chi1 - chi2) / 2.0

#     # First lets construct the phase in the inspiral (region I)
#     phi0 = 1.0
#     phi1 = 0.0
#     phi2 = 3715.0 / 756.0 + 55.0 * eta / 9.0
#     phi3 = (
#         -16.0 * pi
#         + 113.0 * delta * chi_a / 3.0
#         + (113.0 / 3.0 - 76.0 * eta / 3.0) * chi_s
#     )
#     phi4 = (
#         15293365.0 / 508032.0
#         + 27145.0 * eta / 504.0
#         + 3085.0 * (eta ** 2.0) / 72.0
#         + (-405.0 / 8.0 + 200.0 * eta) * (chi_a ** 2.0)
#         - 405.0 * delta * chi_a * chi_s / 4.0
#         + (-405.0 / 8.0 + 5.0 * eta / 2.0) * (chi_s ** 2.0)
#     )
#     phi5 = (1.0 + jnp.log(pi * fM_s)) * (
#         38645.0 * pi / 756.0
#         - 64.0 * pi * eta / 9.0
#         + delta * (-732985.0 / 2268.0 - 140.0 * eta / 9.0) * chi_a
#         + (-732985.0 / 2268.0 + 24260.0 * eta / 81.0 + 340.0 * (eta ** 2.0) / 9.0)
#         * chi_s
#     )
#     phi6 = (
#         11583231236531.0 / 4694215680.0
#         - 6848.0 * EulerGamma / 21.0
#         - 640.0 * (pi ** 2.0) / 3.0
#         + (-15737765635.0 / 3048192.0 + 2255.0 * (pi ** 2.0) / 12.0) * eta
#         + 76055.0 * (eta ** 2.0) / 1728.0
#         - 127825.0 * (eta ** 3.0) / 1296.0
#         - 6848.0 * jnp.log(64.0 * pi * fM_s) / 63.0
#         + 2270.0 * pi * delta * chi_a / 3.0
#         + (2270.0 * pi / 3.0 - 520.0 * pi * eta) * chi_s
#     )
#     phi7 = (
#         77096675.0 * pi / 254016.0
#         + 378515.0 * pi * eta / 1512.0
#         - 74045.0 * pi * (eta ** 2.0) / 756.0
#         + delta
#         * (
#             -25150083775.0 / 3048192.0
#             + 26804935.0 * eta / 6048.0
#             - 1985.0 * (eta ** 2.0) / 48.0
#         )
#         * chi_a
#         + (
#             -25150083775.0 / 3048192.0
#             + 10566655595.0 * eta / 762048.0
#             - 1042165.0 * (eta ** 2.0) / 3024.0
#             + 5345.0 * (eta ** 3.0) / 36.0
#         )
#         * chi_s
#     )

#     # Add frequency dependence here
#     TF2_pre = 3.0 * ((pi * fM_s) ** -(5.0 / 3.0)) / (128.0 * eta)

#     phi_TF2 = TF2_pre * (
#         phi0
#         + phi1 * ((pi * fM_s) ** (1.0 / 3.0))
#         + phi2 * ((pi * fM_s) ** (2.0 / 3.0))
#         + phi3 * ((pi * fM_s) ** (3.0 / 3.0))
#         + phi4 * ((pi * fM_s) ** (4.0 / 3.0))
#         + phi5 * ((pi * fM_s) ** (5.0 / 3.0))
#         + phi6 * ((pi * fM_s) ** (6.0 / 3.0))
#         + phi7 * ((pi * fM_s) ** (7.0 / 3.0))
#     )

#     phi_Ins = (
#         phi_TF2
#         + (
#             coeffs[7] * fM_s
#             + (3.0 / 4.0) * coeffs[8] * (fM_s ** (4.0 / 3.0))
#             + (3.0 / 5.0) * coeffs[9] * (fM_s ** (5.0 / 3.0))
#             + (1.0 / 2.0) * coeffs[10] * (fM_s ** 2.0)
#         )
#         / eta
#     )
#     return phi_Ins


def get_IIa_raw_phase(fM_s: Array, theta: Array, coeffs: Array) -> Array:
    m1, m2, _, _ = theta
    m1_s = m1 * gt
    m2_s = m2 * gt
    M_s = m1_s + m2_s
    eta = m1_s * m2_s / (M_s ** 2.0)

    phi_IIa_raw = (
        coeffs[11] * fM_s
        + coeffs[12] * jnp.log(fM_s)
        - coeffs[13] * (fM_s ** -3.0) / 3.0
    ) / eta

    return phi_IIa_raw


def get_IIb_raw_phase(fM_s: Array, theta: Array, coeffs: Array, f_RD, f_damp) -> Array:
    m1, m2, _, _ = theta
    m1_s = m1 * gt
    m2_s = m2 * gt
    M_s = m1_s + m2_s
    eta = m1_s * m2_s / (M_s ** 2.0)

    f_RDM_s = f_RD * M_s
    f_dampM_s = f_damp * M_s

    phi_IIb_raw = (
        coeffs[14] * fM_s
        - coeffs[15] * (fM_s ** -1.0)
        + 4.0 * coeffs[16] * (fM_s ** (3.0 / 4.0)) / 3.0
        + coeffs[17] * jnp.arctan((fM_s - coeffs[18] * f_RDM_s) / f_dampM_s)
    ) / eta

    return phi_IIb_raw


def get_Amp0(fM_s: Array, eta: float) -> Array:
    Amp0 = (
        (2.0 / 3.0 * eta) ** (1.0 / 2.0) * (fM_s) ** (-7.0 / 6.0) * pi ** (-1.0 / 6.0)
    )
    return Amp0


def get_inspiral_Amp(fM_s: Array, theta: Array, coeffs: Array) -> Array:
    # Below is taken from https://git.ligo.org/lscsoft/lalsuite/-/blob/master/lalsimulation/lib/LALSimIMRPhenomD_internals.c
    # Lines 302 --> 351
    m1, m2, chi1, chi2 = theta
    m1_s = m1 * gt
    m2_s = m2 * gt
    M_s = m1_s + m2_s
    eta = m1_s * m2_s / (M_s ** 2.0)
    eta2 = eta * eta
    eta3 = eta * eta2

    Seta = jnp.sqrt(1.0 - 4.0 * eta)
    SetaPlus1 = 1.0 + Seta

    # Spin variables
    chi12 = chi1 * chi1
    chi22 = chi2 * chi2

    # First lets construct the Amplitude in the inspiral (region I)
    A0 = 1.0
    A2 = ((-969 + 1804 * eta) * pi ** (2.0 / 3.0)) / 672.0
    A3 = (
        (chi1 * (81 * SetaPlus1 - 44 * eta) + chi2 * (81 - 81 * Seta - 44 * eta)) * pi
    ) / 48.0
    A4 = (
        (
            -27312085.0
            - 10287648 * chi22
            - 10287648 * chi12 * SetaPlus1
            + 10287648 * chi22 * Seta
            + 24
            * (-1975055 + 857304 * chi12 - 994896 * chi1 * chi2 + 857304 * chi22)
            * eta
            + 35371056 * eta2
        )
        * (pi ** (4.0 / 3.0))
    ) / 8.128512e6

    A5 = (
        (pi ** (5.0 / 3.0))
        * (
            chi2
            * (-285197 * (-1 + Seta) + 4 * (-91902 + 1579 * Seta) * eta - 35632 * eta2)
            + chi1
            * (285197 * SetaPlus1 - 4 * (91902 + 1579 * Seta) * eta - 35632 * eta2)
            + 42840 * (-1.0 + 4 * eta) * pi
        )
    ) / 32256.0

    A6 = (
        -(
            (pi ** 2.0)
            * (
                -336
                * (
                    -3248849057.0
                    + 2943675504 * chi12
                    - 3339284256 * chi1 * chi2
                    + 2943675504 * chi22
                )
                * eta2
                - 324322727232 * eta3
                - 7
                * (
                    -177520268561
                    + 107414046432 * chi22
                    + 107414046432 * chi12 * SetaPlus1
                    - 107414046432 * chi22 * Seta
                    + 11087290368 * (chi1 + chi2 + chi1 * Seta - chi2 * Seta) * pi
                )
                + 12
                * eta
                * (
                    -545384828789
                    - 176491177632 * chi1 * chi2
                    + 202603761360 * chi22
                    + 77616 * chi12 * (2610335 + 995766 * Seta)
                    - 77287373856 * chi22 * Seta
                    + 5841690624 * (chi1 + chi2) * pi
                    + 21384760320 * (pi ** 2.0)
                )
            )
        )
        / 6.0085960704e10
    )
    A7 = coeffs[0]
    A8 = coeffs[1]
    A9 = coeffs[2]

    Amp_Ins = (
        A0
        # A1 is missed since its zero
        + A2 * (fM_s ** (2.0 / 3.0))
        + A3 * fM_s
        + A4 * (fM_s ** (4.0 / 3.0))
        + A5 * (fM_s ** (5.0 / 3.0))
        + A6 * (fM_s ** 2.0)
        # Now we add the coefficient terms
        + A7 * (fM_s ** (7.0 / 3.0))
        + A8 * (fM_s ** (8.0 / 3.0))
        + A9 * (fM_s ** 3.0)
    )

    return Amp_Ins


def get_IIa_Amp(
    fM_s: Array, theta: Array, coeffs: Array, f1, f3, f_RD, f_damp
) -> Array:
    m1, m2, _, _ = theta
    m1_s = m1 * gt
    m2_s = m2 * gt
    M_s = m1_s + m2_s

    # Central frequency point
    f2 = (f1 + f3) / 2

    # For this region, we also need to calculate the the values and derivatives
    # of the Ins and IIb regions
    v1, d1 = jax.value_and_grad(get_inspiral_Amp)(f1 * M_s, theta, coeffs)
    v3, d3 = jax.value_and_grad(get_IIb_Amp)(f3 * M_s, theta, coeffs, f_RD, f_damp)

    # Here we need the delta solutions
    delta0 = get_delta0(f1 * M_s, f2 * M_s, f3 * M_s, v1, coeffs[3], v3, d1, d3)
    delta1 = get_delta1(f1 * M_s, f2 * M_s, f3 * M_s, v1, coeffs[3], v3, d1, d3)
    delta2 = get_delta2(f1 * M_s, f2 * M_s, f3 * M_s, v1, coeffs[3], v3, d1, d3)
    delta3 = get_delta3(f1 * M_s, f2 * M_s, f3 * M_s, v1, coeffs[3], v3, d1, d3)
    delta4 = get_delta4(f1 * M_s, f2 * M_s, f3 * M_s, v1, coeffs[3], v3, d1, d3)

    Amp_IIa = (
        delta0
        + delta1 * fM_s
        + delta2 * (fM_s ** 2.0)
        + delta3 * (fM_s ** 3.0)
        + delta4 * (fM_s ** 4.0)
    )

    return Amp_IIa


def get_IIb_Amp(fM_s: Array, theta: Array, coeffs: Array, f_RD, f_damp) -> Array:
    m1, m2, _, _ = theta
    m1_s = m1 * gt
    m2_s = m2 * gt
    M_s = m1_s + m2_s
    gamma1 = coeffs[4]
    gamma2 = coeffs[5]
    gamma3 = coeffs[6]
    fDM = f_damp * M_s
    fRD = f_RD * M_s

    fDMgamma3 = fDM * gamma3
    fminfRD = fM_s - fRD
    Amp_IIb = (
        jnp.exp(-(fminfRD) * gamma2 / (fDMgamma3))
        * (fDMgamma3 * gamma1)
        / ((fminfRD) ** 2.0 + (fDMgamma3) ** 2.0)
    )
    return Amp_IIb


@jax.jit
def Phase(f: Array, theta: Array) -> Array:
    """
    Computes the phase of the PhenomD waveform following 1508.07253.
    Sets time and phase of coealence to be zero.

    Returns:
    --------
        phase (array): Phase of the GW as a function of frequency
    """
    # First lets calculate some of the vairables that will be used below
    # Mass variables
    m1, m2, _, _ = theta
    m1_s = m1 * gt
    m2_s = m2 * gt
    M_s = m1_s + m2_s

    coeffs = get_coeffs(theta)

    # Next we need to calculate the transition frequencies
    f1, f2, _, _, f_RD, f_damp = get_transition_frequencies(theta, coeffs[5], coeffs[6])

    phi_Ins = get_inspiral_phase(f * M_s, theta, coeffs)

    # Next lets construct the phase of the late inspiral (region IIa)
    # beta0 is found by matching the phase between the region I and IIa
    # C(1) continuity must be preserved. We therefore need to solve for an additional
    # contribution to beta1
    # Note that derivatives seem to be d/d(fM_s), not d/df

    # Here I've now defined
    # phi_IIa(f1*M_s) + beta0 + beta1_correction*(f1*M_s) = phi_Ins(f1*M_s)
    # ==> phi_IIa'(f1*M_s) + beta1_correction = phi_Ins'(f1*M_s)
    # ==> beta1_correction = phi_Ins'(f1*M_s) - phi_IIa'(f1*M_s)
    # ==> beta0 = phi_Ins(f1*M_s) - phi_IIa(f1*M_s) - beta1_correction*(f1*M_s)
    phi_Ins_f1, dphi_Ins_f1 = jax.value_and_grad(get_inspiral_phase)(
        f1 * M_s, theta, coeffs
    )
    phi_IIa_f1, dphi_IIa_f1 = jax.value_and_grad(get_IIa_raw_phase)(
        f1 * M_s, theta, coeffs
    )

    beta1_correction = dphi_Ins_f1 - dphi_IIa_f1
    beta0 = phi_Ins_f1 - beta1_correction * (f1 * M_s) - phi_IIa_f1

    phi_IIa_func = (
        lambda fM_s: get_IIa_raw_phase(fM_s, theta, coeffs)
        + beta0
        + (beta1_correction * fM_s)
    )
    phi_IIa = phi_IIa_func(f * M_s)

    # And finally, we do the same thing to get the phase of the merger-ringdown (region IIb)
    # phi_IIb(f2*M_s) + a0 + a1_correction*(f2*M_s) = phi_IIa(f2*M_s)
    # ==> phi_IIb'(f2*M_s) + a1_correction = phi_IIa'(f2*M_s)
    # ==> a1_correction = phi_IIa'(f2*M_s) - phi_IIb'(f2*M_s)
    # ==> a0 = phi_IIa(f2*M_s) - phi_IIb(f2*M_s) - beta1_correction*(f2*M_s)
    phi_IIa_f2, dphi_IIa_f2 = jax.value_and_grad(phi_IIa_func)(f2 * M_s)
    phi_IIb_f2, dphi_IIb_f2 = jax.value_and_grad(get_IIb_raw_phase)(
        f2 * M_s, theta, coeffs, f_RD, f_damp
    )

    a1_correction = dphi_IIa_f2 - dphi_IIb_f2
    a0 = phi_IIa_f2 - a1_correction * (f2 * M_s) - phi_IIb_f2

    phi_IIb = (
        get_IIb_raw_phase(f * M_s, theta, coeffs, f_RD, f_damp)
        + a0
        + a1_correction * (f * M_s)
    )

    # And now we can combine them by multiplying by a set of heaviside functions
    phase = (
        phi_Ins * jnp.heaviside(f1 - f, 0.5)
        + jnp.heaviside(f - f1, 0.5) * phi_IIa * jnp.heaviside(f2 - f, 0.5)
        + phi_IIb * jnp.heaviside(f - f2, 0.5)
    )

    return phase


@jax.jit
def Amp(f: Array, theta: Array, D=1) -> Array:
    """
    Computes the amplitude of the PhenomD frequency domain waveform following 1508.07253.
    Note that this waveform also assumes that object one is the more massive.

    Returns:
    --------
      Amplitude (array):
    """

    # First lets calculate some of the vairables that will be used below
    # Mass variables
    m1, m2, _, _ = theta
    m1_s = m1 * gt
    m2_s = m2 * gt
    M_s = m1_s + m2_s
    eta = m1_s * m2_s / (M_s ** 2.0)

    coeffs = get_coeffs(theta)

    _, _, f3, f4, f_RD, f_damp = get_transition_frequencies(theta, coeffs[5], coeffs[6])

    # First we get the inspiral amplitude
    Amp_Ins = get_inspiral_Amp(f * M_s, theta, coeffs)

    # Next lets construct the phase of the late inspiral (region IIa)
    # Note that this part is a little harder since we need to solve a system of equations for deltas
    Amp_IIa = get_IIa_Amp(f * M_s, theta, coeffs, f3, f4, f_RD, f_damp)

    # And finally, we construct the amplitude of the merger-ringdown (region IIb)
    Amp_IIb = get_IIb_Amp(f * M_s, theta, coeffs, f_RD, f_damp)

    # And now we can combine them by multiplying by a set of heaviside functions
    Amp = (
        Amp_Ins * jnp.heaviside(f3 - f, 0.5)
        + jnp.heaviside(f - f3, 0.5) * Amp_IIa * jnp.heaviside(f4 - f, 0.5)
        + Amp_IIb * jnp.heaviside(f - f4, 0.5)
    )

    # Prefactor
    Amp0 = get_Amp0(f * M_s, eta) * (
        2.0 * jnp.sqrt(5.0 / (64.0 * pi))
    )  # This second factor is from lalsuite...

    # Need to add in an overall scaling of M_s^2 to make the units correct
    dist_s = (D * m_per_Mpc) / C
    return Amp0 * Amp * (M_s ** 2.0) / dist_s


@jax.jit
def _gen_IMRPhenomD(
    f: Array, theta_intrinsic: Array, theta_extrinsic: Array, coeffs: Array
):
    M_s = (theta_intrinsic[0] + theta_intrinsic[1]) * gt

    # Shift phase so that peak amplitude matches t = 0
    _, _, _, f4, f_RD, f_damp = get_transition_frequencies(
        theta_intrinsic, coeffs[5], coeffs[6]
    )
    t0 = jax.grad(get_IIb_raw_phase)(f4 * M_s, theta_intrinsic, coeffs, f_RD, f_damp)

    # Lets call the amplitude and phase now
    Psi = Phase(f, theta_intrinsic)
    Mf_ref = f[0] * M_s
    Psi -= t0 * ((f * M_s) - Mf_ref)
    ext_phase_contrib = (
        2.0 * pi * f * theta_extrinsic[1] - theta_extrinsic[2] - pi / 4.0
    )

    Psi += ext_phase_contrib
    A = Amp(f, theta_intrinsic, D=theta_extrinsic[0])

    h0 = A * jnp.exp(1j * -Psi)
    return h0


@jax.jit
def gen_IMRPhenomD(f: Array, params: Array):
    """
    Generate PhenomD frequency domain waveform following 1508.07253.
    Note that this waveform also assumes that object one is the more massive.
    vars array contains both intrinsic and extrinsic variables
    theta = [m1, m2, chi1, chi2, D, tc, phic]
    Mchirp: Chirp mass of the system [solar masses]
    eta: Symmetric mass ratio [between 0.0 and 0.25]
    chi1: Dimensionless aligned spin of the primary object [between -1 and 1]
    chi2: Dimensionless aligned spin of the secondary object [between -1 and 1]
    D: Luminosity distance to source [Mpc]
    tc: Time of coalesence. This only appears as an overall linear in f contribution to the phase
    phic: Phase of coalesence

    Returns:
    --------
      hp (array): Strain of the plus polarization
      hc (array): Strain of the cross polarization
    """
    # Lets make this easier by starting in Mchirp and eta space
    m1, m2 = Mc_eta_to_ms(jnp.array([params[0], params[1]]))
    theta_intrinsic = jnp.array([m1, m2, params[2], params[3]])
    theta_extrinsic = jnp.array([params[4], params[5], params[6]])

    coeffs = get_coeffs(theta_intrinsic)
    h0 = _gen_IMRPhenomD(f, theta_intrinsic, theta_extrinsic, coeffs)
    return h0


@jax.jit
def gen_IMRPhenomD_polar(f: Array, params: Array):
    """
    Generate PhenomD frequency domain waveform following 1508.07253.
    Note that this waveform also assumes that object one is the more massive.
    vars array contains both intrinsic and extrinsic variables
    theta = [m1, m2, chi1, chi2, D, tc, phic]
    Mchirp: Chirp mass of the system [solar masses]
    eta: Symmetric mass ratio [between 0.0 and 0.25]
    chi1: Dimensionless aligned spin of the primary object [between -1 and 1]
    chi2: Dimensionless aligned spin of the secondary object [between -1 and 1]
    D: Luminosity distance to source [Mpc]
    tc: Time of coalesence. This only appears as an overall linear in f contribution to the phase
    phic: Phase of coalesence
    inclination: Inclination angle of the binary
    polarization_angle: Polarization angle of the binary

    Returns:
    --------
      hp (array): Strain of the plus polarization
      hc (array): Strain of the cross polarization
    """
    l, psi = params[7], params[8]
    h0 = gen_IMRPhenomD(f, params)

    hp = h0 * (1 / 2 * (1 + jnp.cos(l) ** 2) * jnp.cos(2 * psi))
    hc = h0 * jnp.cos(l) * jnp.sin(2 * psi)

    return hp, hc