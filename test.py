from formula_utils import str_to_formula

generated_str = '(X p_4)'
target_str = '(X p_3)'
target_formula = str_to_formula(target_str)


try:
    generated_formula = str_to_formula(generated_str)
      
    if str(generated_formula) == str(target_formula):
        print(str(generated_formula) == str(target_formula))

except Exception:
    # Penalize for invalid formula by adding max distance
    print(False)
