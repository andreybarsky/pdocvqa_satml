import argparse
from typing import Optional
from rdp_privacy_calculator import compute_rdp_epsilon
from prv_privacy_calculator import compute_prv_epsilon


def parse_args():
    parser = argparse.ArgumentParser(description="PFL-DocVQA Baseline: Privacy calculator.")
    parser.add_argument(
        "--num_iterations",
        type=int,
        required=True,
        help="Number of iterations rounds.",
    )
    parser.add_argument(
        "--total_number_of_providers_per_client",
        type=int,
        required=True
        help="Number of groups (providers) in the dataset."
    )
    parser.add_argument(
        "--expected_providers_per_iteration",
        type=int,
        required=True,
        help="Number of groups (providers) sampled in each iteration.",
    )
    parser.add_argument(
        "--target_epsilon",
        type=float,
        required=True,
        help="The requested epsilion.",
    )
    parser.add_argument(
        "--accountant",
        type=str,
        choices=["prv", "rdp"],
        default="prv",
        help="Choice between PRV (default) and RDP accountant.",
    )
    parser.add_argument(
        "--expected_number_of_clients_per_round",
        type=int, default=1,
        help="Expected number of clients sampled per round."
    )
    parser.add_argument(
        "--total_number_of_clients",
        type=int, default=1,
        help="Number of clients in the dataset."
    )
    return parser.parse_args()


MAX_SIGMA = 1e6


# taken and modified from pytorch/opacus
def get_noise_multiplier(
    target_epsilon: float,
    target_delta: float,
    sample_rate: float,
    steps: Optional[int] = None,
    epsilon_tolerance: float = 0.01,
    accountant: str = "prv",
) -> float:
    r"""
    Computes the noise level sigma to reach a total budget of (target_epsilon, target_delta)
    at the end of epochs, with a given sample_rate

    Args:
        target_epsilon: the privacy budget's epsilon
        target_delta: the privacy budget's delta
        sample_rate: the sampling rate (usually batch_size / n_data)
        epochs: the number of epochs to run
        steps: number of steps to run
        accountant: accounting mechanism used to estimate epsilon
        epsilon_tolerance: precision for the binary search
    Returns:
        The noise level sigma to ensure privacy budget of (target_epsilon, target_delta)
    """

    eps_high = float("inf")

    sigma_low, sigma_high = 0, 10
    while eps_high > target_epsilon:
        sigma_high = 2 * sigma_high
        if accountant == "prv":
            eps_high = compute_prv_epsilon(
                sigma_high,
                sampling_prob=sample_rate,
                global_iterations=steps,
                delta=target_delta,
            )
        elif accountant == "rdp":
            eps_high = compute_rdp_epsilon(
                sigma_high,
                sampling_prob=sample_rate,
                global_iterations=steps,
                delta=target_delta,
            )
        if sigma_high > MAX_SIGMA:
            raise ValueError("The privacy budget is too low.")

    while target_epsilon - eps_high > epsilon_tolerance:
        sigma = (sigma_low + sigma_high) / 2
        if accountant == "prv":
            eps = compute_prv_epsilon(
                sigma,
                sampling_prob=sample_rate,
                global_iterations=steps,
                delta=target_delta,
            )
        elif accountant == "rdp":
            eps = compute_rdp_epsilon(
                sigma,
                sampling_prob=sample_rate,
                global_iterations=steps,
                delta=target_delta,
            )

        if eps < target_epsilon:
            sigma_high = sigma
            eps_high = eps
        else:
            sigma_low = sigma

    return sigma_high


if __name__ == "__main__":
    args = parse_args()
    sampling_prob = (args.expected_providers_per_iteration / args.total_number_of_providers_per_client)

    # federated learning (just set this ratio to 1 if centralized learning)
    sampling_prob = sampling_prob*(args.expected_number_of_clients_per_round / args.total_number_of_clients)
    DELTA = 10**-5  # DP parameter; we fix delta and compute the minimum epsilon

    noise_multiplier = get_noise_multiplier(
        target_epsilon=args.target_epsilon,
        target_delta=DELTA,
        sample_rate=sampling_prob,
        steps=args.num_iterations,
        accountant=args.accountant,
    )

    if args.accountant == "prv":
        eps = compute_prv_epsilon(
            sigma=noise_multiplier,
            sampling_prob=sampling_prob,
            global_iterations=args.num_iterations,
            delta=DELTA,
        )
    elif args.accountant == "rdp":
        eps = compute_rdp_epsilon(
            sigma=noise_multiplier,
            sampling_prob=sampling_prob,
            global_iterations=args.num_iterations,
            delta=DELTA,
        )
    print(f"{args.accountant} accountant: target epsilon {args.target_epsilon} noise_multiplier {noise_multiplier} achieved epsilon {eps}")
