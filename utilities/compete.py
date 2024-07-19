import math

MAX_PENALTY = 0.03
MAX_PENALTY_SCALE = 100
PENALTY_THRESHOLD = 0.96


def calculate_penalty(block_one: int, block_two: int) -> float:
    """
    Calculate an exponentially increasing penalty based on the difference between two epoch steps.
    The closer in time difference, the smaller the penalty
    The larger in time difference (up to 100 blocks), the larger the penalty
    Returns:
    float: Penalty value between 0 and MAX_PENALTY
    """
    step_difference = abs(block_one - block_two)

    if step_difference >= MAX_PENALTY_SCALE:
        return MAX_PENALTY

    # Exponential penalty = max_penalty * (1 - e^(-k * step_difference))
    # Solve for k: max_penalty * (1 - e^(-k * scale_factor)) = max_penalty
    k = -math.log(1e-6) / MAX_PENALTY_SCALE  # Using 1e-6 to avoid division by zero

    penalty = MAX_PENALTY * (1 - math.exp(-k * step_difference))
    return penalty


def iswin(score_i, score_j, block_i, block_j):
    """
    Determines the winner between two models based on the epsilon adjusted loss.

    Parameters:
        loss_i (float): Loss of uid i on batch
        loss_j (float): Loss of uid j on batch.
        block_i (int): Block of uid i.
        block_j (int): Block of uid j.
    Returns:
        bool: True if loss i is better, False otherwise.
    """
    penalty = MAX_PENALTY

    # Adjust score based on timestamp and pretrain epsilon
    score_i = (1 - penalty) * score_i if block_i > block_j else score_i
    score_j = (1 - penalty) * score_j if block_j > block_i else score_j
    return score_i > score_j
