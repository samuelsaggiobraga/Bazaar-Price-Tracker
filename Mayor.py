from mayor_utils import get_mayor_perks

# Fetch all mayor data
mayor_data = get_mayor_perks()

print(f"Loaded {len(mayor_data)} mayor periods:")
for i, mayor in enumerate(mayor_data):
    active_perks = sum(mayor['perks'])
    print(f"Mayor {i+1}: {mayor['start_date'].strftime('%Y-%m-%d')} - {active_perks} active perks")

    

    



    

