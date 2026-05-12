
import re


def format_floats_to_sig_figs(s, sig_figs=3):
    def format_match(match):
        num = float(match.group())
        # Format to specified significant figures
        formatted = f"{num:.{sig_figs - 1}e}"  # Scientific notation
        # Parse the scientific notation to get properly formatted number
        mantissa, exponent = formatted.split('e')
        exp_val = int(exponent)

        # Convert back to decimal notation with appropriate precision
        if exp_val >= 0:
            # Positive exponent
            result = f"{float(mantissa) * 10 ** exp_val:.{max(0, sig_figs - exp_val - 1)}f}"
        else:
            # Negative exponent
            result = f"{float(mantissa) * 10 ** exp_val:.{sig_figs - 1 + abs(exp_val)}f}"

        # Remove trailing zeros after decimal point, but keep the decimal if needed
        if '.' in result:
            result = result.rstrip('0').rstrip('.')

        return result

    # Match floating point numbers (including negative)
    pattern = r'-?\d+\.\d+'
    return re.sub(pattern, format_match, s)
