from prv_accountant.dpsgd import DPSGDAccountant
import argparse

def compute_prv_epsilon(sigma, sampling_prob, global_iterations, delta):
    prv_accountant = DPSGDAccountant(
        noise_multiplier=sigma,
        sampling_probability=sampling_prob,
        max_steps=global_iterations,
        eps_error=0.01,
        delta_error=1e-9,
    )
    _, _, eps = prv_accountant.compute_epsilon(num_steps=global_iterations, delta=delta)
    return float(eps)


def parse_args():
    parser = argparse.ArgumentParser(description="PFL-DocVQA: Centralized privacy calculator.")

    # Required
    parser.add_argument("--noise_multiplier", type=float, required=True, help="Noise multiplier.")
    parser.add_argument("--num_iterations", type=int, required=True, help="Number of steps.")
    parser.add_argument(
        "--providers_per_iteration", type=int, required=True, help="Number of groups (providers) sampled in each step."
    )
    parser.add_argument(
        "--num_total_providers_in_dataset", type=int, default=4149, help="Number of groups (providers) in the dataset."
    )
    parser.add_argument("--delta", type=float, default=1e-5, help="Delta for privacy analysis")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    sampling_prob = args.providers_per_iteration / args.num_total_providers_in_dataset

    epsilon = compute_prv_epsilon(
        sigma=args.noise_multiplier, sampling_prob=sampling_prob, global_iterations=args.num_iterations, delta=args.delta
    )
    print(
        f"The privacy parameters result in a epsilon of {epsilon} at delta {args.delta} when using the PRV accountant."
    )
    print(
        f"(noise_multiplier {args.noise_multiplier}, num_steps {args.num_iterations}, num_providers {args.providers_per_iteration})"
    )
