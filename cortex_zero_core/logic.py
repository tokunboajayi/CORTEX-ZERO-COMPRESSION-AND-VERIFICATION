import math
from .models import Opinion

def discount_opinion(opinion: Opinion, reliability: float) -> Opinion:
    """
    Jøsang's Discount Operator (Operator: :sub:`R`)
    Scales belief and disbelief by the reliability of the source.
    Uncertainty increases to fill the gap.
    
    Formula:
    b_new = reliability * b
    d_new = reliability * d
    u_new = 1 - (b_new + d_new)
    """
    b_new = reliability * opinion.belief
    d_new = reliability * opinion.disbelief
    u_new = 1.0 - (b_new + d_new)
    
    # Ensure float precision safety
    if u_new < 0.0: u_new = 0.0
    if u_new > 1.0: u_new = 1.0
    
    return Opinion(
        belief=b_new, 
        disbelief=d_new, 
        uncertainty=u_new, 
        base_rate=opinion.base_rate
    )

def consensus_opinion(op_a: Opinion, op_b: Opinion) -> Opinion:
    """
    Jøsang's Cumulative Consensus Operator (Operator: +)
    Fuses two opinions into a single dogmatic or vacant opinion.
    
    Handles the edge case of K=0 (Dogmatic Conflict) by returning a max-uncertainty opinion
    to indicate total contradiction/confusion, or a safe fallback.
    For this MVP, we return a Vacuous opinion (0, 0, 1) on hard conflict.
    """
    # Calculate divisor K
    # K = u_a + u_b - u_a * u_b
    k = op_a.uncertainty + op_b.uncertainty - (op_a.uncertainty * op_b.uncertainty)
    
    if abs(k) < 1e-9:
        # K is zero. This happens if both u_a and u_b are 0.
        # This is a dogmatic situation.
        # If they agree (b=1, b=1), it's fine. If they disagree (b=1, d=1), it's a conflict.
        # Average the beliefs to handle conflict gracefully in this MVP.
        # Strictly speaking, Jøsang says this is undefined or handled by other operators.
        
        # Simple Conflict Resolution: Average
        new_b = (op_a.belief + op_b.belief) / 2
        new_d = (op_a.disbelief + op_b.disbelief) / 2
        new_u = (op_a.uncertainty + op_b.uncertainty) / 2
        
        # Normalize just in case
        total = new_b + new_d + new_u
        if total > 0:
            return Opinion(belief=new_b/total, disbelief=new_d/total, uncertainty=new_u/total, base_rate=op_a.base_rate)
        else:
            return Opinion(belief=0, disbelief=0, uncertainty=1, base_rate=op_a.base_rate)

    # Standard Case
    b_new = (op_a.belief * op_b.uncertainty + op_b.belief * op_a.uncertainty) / k
    d_new = (op_a.disbelief * op_b.uncertainty + op_b.disbelief * op_a.uncertainty) / k
    u_new = (op_a.uncertainty * op_b.uncertainty) / k
    
    # Fusion of base rates (usually just simple average or weighted, kept simple here as a_a)
    # The prompt didn't specify base rate fusion, keeping a_a or a_b is standard if they match.
    # We will use the base rate from op_a for continuity.
    
    # Cap values to valid range due to float drift
    b_new = min(max(b_new, 0.0), 1.0)
    d_new = min(max(d_new, 0.0), 1.0)
    u_new = min(max(u_new, 0.0), 1.0)
    
    # Re-normalize u to ensure sum 1.0 strict
    # Sometimes b+d+u might be 1.000000001
    remaining = 1.0 - (b_new + d_new)
    if remaining < 0: remaining = 0
    u_new = remaining
    
    return Opinion(belief=b_new, disbelief=d_new, uncertainty=u_new, base_rate=op_a.base_rate)

def calculate_ess(opinion: Opinion, age: float, volatility: float) -> float:
    """
    Epistemic Stability Score (ESS) Formula
    ESS = E + T - V
    """
    # Step 1: Probability Expectation E = b + (a * u)
    e = opinion.belief + (opinion.base_rate * opinion.uncertainty)
    
    # Step 2: Time Bonus T = 0.1 * log(1 + age)
    # Natural log or log10? Usually ln. distinct "log" implies ln in python math.
    t_bonus = 0.1 * math.log(1 + age)
    
    # Step 3: Volatility Penalty V = 0.1 * volatility
    v_penalty = 0.1 * volatility
    
    # Final Calc
    ess = e + t_bonus - v_penalty
    
    # Clamp
    if ess < 0.0: ess = 0.0
    if ess > 1.0: ess = 1.0
    
    return ess
