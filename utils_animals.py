ANIMAL_RU = {
    "Wolf": {"male": "Волк", "female": "Волчица"},
    "Lion": {"male": "Лев", "female": "Львица"},
    "Tiger": {"male": "Тигр", "female": "Тигрица"},
    "Lynx": {"male": "Рысь", "female": "Рысь"},
    "Panther": {"male": "Пантера", "female": "Пантера"},
    "Bear": {"male": "Медведь", "female": "Медведица"},
    "Fox": {"male": "Койот", "female": "Лиса"},
    "Wolverine": {"male": "Росомаха", "female": "Росомаха"},
    "Deer": {"male": "Олень", "female": "Лань"},
    "Monkey": {"male": "Обезьяна", "female": "Обезьяна"},
    "Rabbit": {"male": "Кролик", "female": "Кролик"},
    "Buffalo": {"male": "Буйвол", "female": "Буйволица"},
    "Ram": {"male": "Баран", "female": "Ибекса"},
    "Capybara": {"male": "Капибара", "female": "Капибара"},
    "Elephant": {"male": "Слон", "female": "Слониха"},
    "Horse": {"male": "Конь", "female": "Лошадь"},
    "Eagle": {"male": "Орёл", "female": "Орлица"},
    "Owl": {"male": "Филин", "female": "Сова"},
    "Raven": {"male": "Ворон", "female": "Ворона"},
    "Parrot": {"male": "Попугай", "female": "Попугаиха"},
    "Snake": {"male": "Змей", "female": "Змея"},
    "Crocodile": {"male": "Крокодил", "female": "Крокодил"},
    "Turtle": {"male": "Черепаха", "female": "Черепаха"},
    "Lizard": {"male": "Ящерица", "female": "Ящерица"},
}


def get_animal_ru_name(animal_code: str, gender: str) -> str:
    g = "female" if gender == "female" else "male"
    return ANIMAL_RU[animal_code][g]


ANIMAL_DISPLAY = {
    "en": {
        "Wolf": "Wolf",
        "Lion": "Lion",
        "Tiger": "Tiger",
        "Lynx": "Lynx",
        "Panther": "Panther",
        "Bear": "Bear",
        "Fox": "Fox",
        "Wolverine": "Wolverine",
        "Deer": "Deer",
        "Monkey": "Monkey",
        "Rabbit": "Rabbit",
        "Buffalo": "Buffalo",
        "Ram": "Ram",
        "Capybara": "Capybara",
        "Elephant": "Elephant",
        "Horse": "Horse",
        "Eagle": "Eagle",
        "Owl": "Owl",
        "Raven": "Raven",
        "Parrot": "Parrot",
        "Snake": "Snake",
        "Crocodile": "Crocodile",
        "Turtle": "Turtle",
        "Lizard": "Lizard",
    },
    "es": {
        "Wolf": "Lobo",
        "Lion": "León",
        "Tiger": "Tigre",
        "Lynx": "Lince",
        "Panther": "Pantera",
        "Bear": "Oso",
        "Fox": "Zorro",
        "Wolverine": "Glotón",
        "Deer": "Ciervo",
        "Monkey": "Mono",
        "Rabbit": "Conejo",
        "Buffalo": "Búfalo",
        "Ram": "Carnero",
        "Capybara": "Capibara",
        "Elephant": "Elefante",
        "Horse": "Caballo",
        "Eagle": "Águila",
        "Owl": "Búho",
        "Raven": "Cuervo",
        "Parrot": "Loro",
        "Snake": "Serpiente",
        "Crocodile": "Cocodrilo",
        "Turtle": "Tortuga",
        "Lizard": "Lagarto",
    },
    "pt": {
        "Wolf": "Lobo",
        "Lion": "Leão",
        "Tiger": "Tigre",
        "Lynx": "Lince",
        "Panther": "Pantera",
        "Bear": "Urso",
        "Fox": "Raposa",
        "Wolverine": "Carcaju",
        "Deer": "Cervo",
        "Monkey": "Macaco",
        "Rabbit": "Coelho",
        "Buffalo": "Búfalo",
        "Ram": "Carneiro",
        "Capybara": "Capivara",
        "Elephant": "Elefante",
        "Horse": "Cavalo",
        "Eagle": "Águia",
        "Owl": "Coruja",
        "Raven": "Corvo",
        "Parrot": "Papagaio",
        "Snake": "Serpente",
        "Crocodile": "Crocodilo",
        "Turtle": "Tartaruga",
        "Lizard": "Lagarto",
    },
}


def get_animal_display_name(animal_code: str, lang: str, gender: str) -> str:
    if lang == "ru":
        return get_animal_ru_name(animal_code, gender)
    if lang not in ANIMAL_DISPLAY:
        return get_animal_ru_name(animal_code, gender)
    return ANIMAL_DISPLAY[lang].get(animal_code, animal_code)


ELEMENT_NUMBER = {
    "Воздух": 1,
    "Вода": 2,
    "Огонь": 3,
    "Земля": 4,
}

ELEMENT_LABELS = {
    "ru": {
        "Воздух": "Воздуха",
        "Вода": "Воды",
        "Огонь": "Огня",
        "Земля": "Земли",
    },
    "en": {
        "Воздух": "Air",
        "Вода": "Water",
        "Огонь": "Fire",
        "Земля": "Earth",
    },
    "es": {
        "Воздух": "Aire",
        "Вода": "Agua",
        "Огонь": "Fuego",
        "Земля": "Tierra",
    },
    "pt": {
        "Воздух": "Ar",
        "Вода": "Água",
        "Огонь": "Fogo",
        "Земля": "Terra",
    },
}


def get_element_display_name(
    element_code: str, lang: str, ru_case: str | None = None
) -> str:
    if lang == "ru":
        if ru_case == "genitive_for_archetype_line":
            return ELEMENT_LABELS["ru"].get(element_code, element_code)
        return element_code
    labels = ELEMENT_LABELS.get(lang, ELEMENT_LABELS["ru"])
    return labels.get(element_code, element_code)


FEMALE_SUFFIX_ANIMALS = {"Deer", "Fox", "Lion", "Ram"}


def build_image_key(animal_code: str, element: str, gender: str) -> str:
    element_number = ELEMENT_NUMBER.get(element)
    if not element_number:
        raise ValueError(f"Unsupported element for image key: {element}")
    suffix = "_f" if gender == "female" and animal_code in FEMALE_SUFFIX_ANIMALS else ""
    return f"{animal_code}{element_number}{suffix}"
