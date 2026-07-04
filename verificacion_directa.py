"""
Script de verificación que el usuario debe correr EN SU TERMINAL,
apuntando a SU archivo real (C:\\Users\\Usuario\\Desktop\\fuol\\ai_web_agent.py),
no a ninguna copia. No requiere GEMINI_API_KEY ni conexión a internet:
solo lee el texto del archivo y lo analiza.
"""
import ast
import sys
 
path = sys.argv[1] if len(sys.argv) > 1 else "ai_web_agent.py"
 
with open(path, encoding="utf-8") as f:
    src = f.read()
 
print(f"Analizando: {path}\n")
 
# Chequeo 1: cuántas veces aparece 'return response.text' literal
# (esto es un vestigio del bug original; en la versión corregida esta línea
# exacta no debería estar, porque se reemplazó por 'text = response.text'
# seguido de lógica adicional y luego 'return text')
count_return_response = src.count("return response.text")
print(f"1) Apariciones literales de 'return response.text': {count_return_response}")
if count_return_response >= 1:
    print("   ❌ Esta línea es el bug original. Si aparece, el return-duplicado sigue ahí.")
else:
    print("   ✅ No aparece -- consistente con la versión corregida (usa 'return text').")
 
# Chequeo 2: código inalcanzable vía AST (el chequeo real, no de texto)
tree = ast.parse(src)
 
def check_unreachable(body, path_label=""):
    seen_return = False
    issues = []
    for stmt in body:
        if seen_return:
            issues.append(f"{path_label} linea {stmt.lineno}")
        if isinstance(stmt, ast.Return):
            seen_return = True
    return issues
 
all_issues = []
for node in ast.walk(tree):
    if isinstance(node, ast.FunctionDef):
        all_issues += check_unreachable(node.body, node.name)
        for child in ast.walk(node):
            if isinstance(child, ast.Try):
                all_issues += check_unreachable(child.body, f"{node.name}/try")
 
print(f"\n2) Código inalcanzable (AST): {'❌ ' + str(all_issues) if all_issues else '✅ ninguno'}")
 
# Chequeo 3: dentro de explain_prediction, ¿el 'if missing' está ANTES o DESPUÉS del return final?
for node in ast.walk(tree):
    if isinstance(node, ast.FunctionDef) and node.name == "explain_prediction":
        for child in ast.walk(node):
            if isinstance(child, ast.Try):
                lines_in_try = [ast.dump(s)[:40] for s in child.body]
                return_lines = [s.lineno for s in child.body if isinstance(s, ast.Return)]
                if_lines = [s.lineno for s in child.body if isinstance(s, ast.If)]
                print(f"\n3) Dentro del try de explain_prediction:")
                print(f"   líneas de 'return': {return_lines}")
                print(f"   líneas de 'if': {if_lines}")
                if return_lines and if_lines and min(return_lines) < max(if_lines):
                    print("   ❌ Hay un 'return' antes de un 'if' en el mismo bloque -> el if nunca se alcanza")
                elif return_lines and if_lines:
                    print("   ✅ El 'if' aparece antes que el 'return' -> orden correcto")
 
# Chequeo 4: rango del guardrail en código real (usando AST, no texto)
# Esto es más confiable que buscar "0.80" en el texto crudo, porque un
# docstring o comentario que MENCIONE el rango viejo a modo explicativo
# no cuenta como bug -- lo que importa es si aparece dentro de una llamada
# real a max()/min() en el código ejecutable.
found_in_executable_code = []
for node in ast.walk(tree):
    if isinstance(node, ast.Constant) and isinstance(node.value, float):
        if node.value in (0.80, 1.20):
            found_in_executable_code.append((node.lineno, node.value))
 
if found_in_executable_code:
    print(f"\n4) Rango viejo en CÓDIGO EJECUTABLE (no comentarios): ❌ SÍ aparece -> {found_in_executable_code}")
else:
    print(f"\n4) Rango viejo en CÓDIGO EJECUTABLE (no comentarios): ✅ no aparece")
    print("   (puede seguir mencionado en comentarios/docstrings como explicación histórica -- eso es correcto y esperado)")
