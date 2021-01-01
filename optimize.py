import json

import pandas as pd
import pulp
from pulp import lpSum, LpProblem, LpMaximize, LpVariable, LpStatus
from tabulate import tabulate


class NutritionVector:
    def __init__(self, fat, carbs, protein):
        self.fat = fat
        self.carbs = carbs
        self.protein = protein

    @property
    def calories(self):
        return 9 * self.fat + 4 * self.carbs + 4 * self.protein

    def scale(self, factor):
        return self.__class__(self.fat * factor, self.carbs * factor, self.protein * factor)


class Product:
    def __init__(self, name, serving_size, servings_available, nutrients):
        self.name = name
        self.serving_size = serving_size
        self.servings_available = servings_available
        self.nutrition = NutritionVector(**nutrients).scale(1/serving_size)

    @classmethod
    def load_from_json(cls, fp):
        return [cls(**p) for p in json.load(fp)]


def optimize(products, target_macros):
    fat_coefs = dict()
    carb_coefs = dict()
    protein_coefs = dict()
    grams_available = dict()

    for p in products:
        fat_coefs[p.name] = p.nutrition.fat
        carb_coefs[p.name] = p.nutrition.carbs
        protein_coefs[p.name] = p.nutrition.protein
        grams_available[p.name] = p.serving_size * p.servings_available

    mixture_vars = LpVariable.dicts('Mixture', [p.name for p in products], 0)

    fat_sum = lpSum([fat_coefs[p.name] * mixture_vars[p.name] for p in products])
    carb_sum = lpSum([carb_coefs[p.name] * mixture_vars[p.name] for p in products])
    protein_sum = lpSum([protein_coefs[p.name] * mixture_vars[p.name] for p in products])

    prob = LpProblem('Macro Blending', LpMaximize)
    prob += fat_sum == target_macros.fat, 'Fat target'
    prob += carb_sum == target_macros.carbs, 'Carbs target'
    prob += protein_sum == target_macros.protein, 'Protein target'

    for p in products:
        max_mixture = grams_available[p.name]
        min_mixture = 0
        prob += mixture_vars[p.name] <= max_mixture, f'{max_mixture} gram upper bound for {p.name}'
        prob += mixture_vars[p.name] >= min_mixture, f'{min_mixture} gram lower bound for {p.name}'

    prob.solve()
    print('Status:', LpStatus[prob.status])

    records = []
    for p in products:
        grams = mixture_vars[p.name].varValue
        servings = grams / p.serving_size
        fats = p.nutrition.scale(grams).fat
        carbs = p.nutrition.scale(grams).carbs
        proteins = p.nutrition.scale(grams).protein
        cals = p.nutrition.scale(grams).calories
        records.append({'Product': p.name, 'Grams': grams, 'Servings': servings,
                        'Fat (g)': fats,
                        'Carbs (g)': carbs,
                        'Protein (g)': proteins,
                        'Calories': cals})

    print('Eat this:')
    df = pd.DataFrame.from_records(records, columns=['Grams', 'Product', 'Servings',
                                                     'Fat (g)', 'Carbs (g)', 'Protein (g)', 'Calories'])
    df_filtered = df[lambda x: x['Grams'] > 0].round(2)
    print(tabulate(df_filtered, headers='keys', tablefmt='psql', showindex=False))
    print('')

    print('Summary:')
    summary_nutrition = NutritionVector(*map(pulp.value, [fat_sum, carb_sum, protein_sum]))
    df_summary = pd.DataFrame.from_records([
        {'Macronutrient': 'Fat', 'Grams': summary_nutrition.fat, 'Targeted': target_macros.fat},
        {'Macronutrient': 'Carbs', 'Grams': summary_nutrition.carbs, 'Targeted': target_macros.carbs},
        {'Macronutrient': 'Protein', 'Grams': summary_nutrition.protein, 'Targeted': target_macros.protein},
        {'Macronutrient': 'Total calories', 'Grams': summary_nutrition.calories, 'Targeted': target_macros.calories},
    ])
    df_summary = df_summary.set_index('Macronutrient')[['Grams', 'Targeted']].round(2)
    print(tabulate(df_summary, headers='keys', tablefmt='psql', showindex=True))


def main():
    products = Product.load_from_json(open('inventory.json'))
    target_macros = NutritionVector(**json.load(open('objective.json')))
    optimize(products, target_macros)


if __name__ == "__main__":
    main()
