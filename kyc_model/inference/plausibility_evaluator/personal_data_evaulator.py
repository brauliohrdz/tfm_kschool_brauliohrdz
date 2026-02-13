import json

from kyc_model.inference.llm_model_handler import LLMModel


def _generate_prompt_for_name(name: str) -> str:
    base_prompt = """
        TAREA: Evalúa si un nombre completo parece real y típico de una persona española.

        ENTRADA: un string con nombre y apellidos (normalmente 2 o 3 palabras; a veces 4 por nombres compuestos).

        SALIDA: Devuelve SOLO un JSON válido, sin texto adicional y sin formatear:
        {"puntuacion": <int 0..10>, "razones": ["...", "..."]}

        REGLA DE PRIORIDAD (anula todo):
        1) Si el nombre o la combinación coincide o suena claramente a personaje de ficción, celebridad muy conocida, marca/empresa o meme -> puntuacion 0..2.

        CRITERIO PRINCIPAL (si NO aplica la regla de prioridad):
        Evalúa “común en España” vs “común en hispanohablantes pero no típico de España” vs “poco común/no hispano”.

        RÚBRICA DE PUNTUACIÓN:
        - 9..10: Nombre (pila) común en España Y dos apellidos muy comunes en España.
        - 7..8: Suena español de España y plausible; al menos 2 de 3 partes (nombre, apellido1, apellido2) son comunes en España o Latinoamérica.
        - 5..6: Suena hispano/latino y plausible, pero no especialmente típico de España o hay incertidumbre razonable.
        - 3..4: Mezcla rara (p.ej. nombre hispano + apellidos no hispanos) o estructura poco típica; parece posible pero poco probable en España.
        - 0..2: No hispano evidente, texto raro/no nombre humano, o sospecha fuerte de famoso/ficción (ver prioridad).
        - 0 : Ni el nombre ni los apellidos son hispanos.

        HEURÍSTICAS (úsalas para decidir sin inventar datos):
        - España: apellidos muy típicos suelen ser de patrón castellano (ej. terminaciones -ez como Pérez, González, Hernández; y apellidos comunes como García, Martínez, López, Sánchez, Rodríguez, Fernández, Gómez, Díaz, Ruiz, Moreno, Muñoz, Jiménez, Álvarez, Romero, Alonso, Gutiérrez, Navarro, Torres).
        - Ignora mayúsculas y minúsculas.
        - Nombre común España: José, Antonio, Braulio, Manuel, Francisco, David, Javier, Juan, Carlos, Sergio, Daniel, Pablo, Alejandro, Alberto, Rubén, Raúl; y femeninos como María, Carmen, Ana, Laura, Sara, Marta, Lucía, Cristina, Paula, Elena, Patricia, Josefa, Pilar, Monica, Paula, Nerea, Beatriz, Irene, 
        LUIS, VERA, LUNA, VICTORIA, ROBERTO, JOSÉ, MANUELA, ZOE, HUGO, LEO, MARCOS, JUAN, MANUELA, PAOLA, SOFÍA, MARA, MIGUEL, FRANCISCO, POL, ALONSO, ANDRÉS, ORIOL, LAIA, NATALIA, ALBERTO, NUAZET, FÁTIMA, ALEJANDRA.
        - Hispano no España: nombres/apellidos muy frecuentes en LatAm pero menos típicos en España suben a 5..6 (si todo sigue siendo claramente hispano).
        - No hispano: grafías/sonidos claramente ajenos (p.ej. apellidos anglos, eslavos, chinos, etc.) tienden a 0..4 según mezcla.
        - Estructura: 1 palabra o tokens raros (números, emojis, muchas iniciales) baja puntuación.
        - Baja fiabilidad: ni el nombre ni los apellidos son comunes en españa. la puntuación estará entre 0..1
        - Alta fiabilidad: Ambos apellidos son comunes en españa, la puntuación debe estar por encima de 6
        -Apellidos comunes en España (ejemplos): García, González, Rodríguez, Fernández, López, Martínez, Sánchez, Pérez, Gómez, Martín, Jiménez, Ruiz, Hernández, Díaz, Moreno, Muñoz, Álvarez, Romero, Alonso, Gutiérrez, Navarro, Torres, Domínguez, Vázquez,
        Ramos, Gil, Ramírez, Serrano, Blanco, Molina, Suárez, Ortega, Delgado, Castro, Ortiz, Marín, Rubio, Sanz, Núñez, Medina, Iglesias, Garrido, 
        Cortés, Santos, Castillo, Guerrero, Lozano, Cano, Prieto, Méndez, Calvo, Cruz, Herrero, Gallego, León, Márquez, Peña, Flores, Campos, Cabrera, 
        Fuentes, Carrasco, Díez, Rey, Vega, Rojas, Pastor, Nieto, Aguilar, Arias, Vicente, Benítez, Crespo, Lorenzo, Soler, Paredes, Moya, Esteban, Parra, 
        Ferrer, Giménez, Ibáñez, Salvador, Serra, Miró, Durán,Pita, Pons, Puig, Riera, Costa, Solé, Pascual, Blasco, Segura, Cervera, Bernal.
        LUIS, VERA, LUNA, VICTORIA, ROBERTO, JOSÉ, MANUELA, ZOE, HUGO, LEO, MARCOS, JUAN, MANUELA, PAOLA, SOFÍA, MARA, MIGUEL, FRANCISCO, POL, ALONSO, 
        ANDRÉS, ORIOL, LAIA, NATALIA, ALBERTO, NUAZET, FÁTIMA, ALEJANDRA

        RESTRICCIONES:
        - "score" debe ser ENTERO 0..10.
        - "reason": 1 a 3 frases cortas y concretas (máx 12 palabras cada una).
        - No añadas claves extra. No incluyas markdown. No incluyas explicación fuera del JSON.

        EJEMPLOS (guía de consistencia):
        Entrada: "María García López"
        Salida: {"score":10,"reason":["Nombre y apellidos muy comunes en España."]}

        Entrada: "James Pérez Ramos"
        Salida: {"score":8,"reason":["Nombre poco común en España.", "Ambos apellidos son comunes en España.", "Estructura poco típica de España."]}

        Entrada: "Brayan Yeferson Caicedo"
        Salida: {"score":6,"reason":["Suena hispano/latino.","No es especialmente típico en España."]}

        Entrada: "Harry Potter"
        Salida: {"score":1,"reason":["Personaje de ficción muy conocido."]}
        """

    prompt = base_prompt + "\n\nNOMBRE_A_EVALUAR: " + name
    return prompt


def score_name_autheticity(full_name: str, llm_model: LLMModel) -> dict:

    prompt = _generate_prompt_for_name(full_name)
    llm_response = llm_model.ask(prompt)
    try:
        llm_response = llm_response.replace("```json", "").replace("```", "").replace("\n", "")
        return json.loads(llm_response)
    except ValueError as e:
        print(str(e), llm_response)
        return {"score": None, "reason": ["Error al evaluar el nombre"]}
