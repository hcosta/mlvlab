# **Arquitectura Sistema Algoritmos MLV-Lab**

La filosofía de este sistema es la **separación de responsabilidades**. Cada componente tiene un trabajo muy específico, lo que hace que el framework sea flexible, robusto y, lo más importante, fácil de extender.

## **1\. El Lanzador (The Launcher): mlvlab/cli/main.py 🚀**

Este es el **punto de entrada principal** que ve el usuario en la terminal.

* **Responsabilidad:** Su único trabajo es interpretar los comandos (play, train, eval) y los argumentos que le pasas (como el env\_id).  
* **NO contiene lógica de juego ni de IA.** En lugar de hacer el trabajo él mismo, **lanza un nuevo proceso de Python** ejecutando un script "ayudante" (como train\_entry.py o player\_entry.py).  
* **¿Por qué?** Esto mantiene la interfaz de línea de comandos (CLI) limpia y rápida, y aísla cada tarea en su propio proceso, evitando conflictos de librerías o estados.

## **2\. Los Puntos de Entrada (The Entry Points): mlvlab/cli/\*\_entry.py 🚪**

Estos son los **guiones intermediarios** que el lanzador ejecuta. Son el puente entre la petición del usuario y la lógica real del algoritmo.

* **Responsabilidad:** Toman los argumentos de la línea de comandos, cargan la configuración inicial del entorno (get\_env\_config), y preparan todo lo necesario para llamar al algoritmo correcto.  
* train\_entry.py, por ejemplo, carga el config.py del entorno para saber qué algoritmo usar (ALGORITHM \= "ql" o "random"), prepara el directorio para guardar los resultados (run\_dir), y finalmente llama al método train del plugin correspondiente.

## **3\. El Corazón del Sistema: El Framework de Plugins (mlvlab/algorithms/) ❤️**

Esta es la parte más importante y la que hace que el sistema sea extensible.

#### **El Contrato (registry.py)**

El archivo registry.py define el protocolo AlgorithmPlugin. Esto es como un **contrato** que obliga a que todos los algoritmos tengan exactamente los mismos métodos (key, build\_agent, train, eval). Esto garantiza que train\_entry.py pueda llamar a algo.train(...) sin importar si algo es Q-Learning, un Agente Aleatorio o DQN.

También contiene el **registro central \_ALGORITHMS**, que es como una agenda telefónica. Cada plugin, al ser importado, se apunta en esta agenda usando su key.

#### **La Detección Automática (\_\_init\_\_.py)**

Este archivo es el "mago" del sistema. En lugar de tener que importar manualmente cada plugin nuevo que crees, este script **escanea automáticamente** todas las subcarpetas dentro de mlvlab/algorithms/.

Si encuentra una carpeta que contiene un archivo plugin.py, lo importa. El propio plugin.py se auto-registra al ser importado (gracias a la línea register\_algorithm(...) al final del archivo).

**Resultado:** Para añadir un nuevo algoritmo, solo tienes que crear su carpeta y su plugin.py. ¡No necesitas tocar ningún otro archivo de la estructura central\!

#### **Los Plugins (ql/plugin.py, random/plugin.py, etc.)**

Estos son los **trabajadores reales**. Cada plugin es una caja negra autosuficiente que contiene toda la lógica específica de su algoritmo.

* El QLearningPlugin, por ejemplo, sabe que necesita un state\_adapter, sabe cómo construir un QLearningAgent, sabe cómo ejecutar el bucle de entrenamiento learn, y sabe cómo guardar la q\_table.  
* train\_entry.py no sabe nada de esto, simplemente le dice: "Oye, Q-Learning, entrénate con esta configuración".

## **4\. Relación entre AlgorithmPlugin y Agent 🧑‍🏫🧠**

Esta es la distinción más importante de la arquitectura. En resumen: el **AlgorithmPlugin** es el **entrenador**, mientras que el **Agent** es el **aprendiz** o el **cerebro**.

#### **Una Analogía: El Entrenador y el Atleta**

* El **Agent** (ej: QLearningAgent) es como un **atleta**. Tiene el conocimiento específico de su disciplina: sabe cómo ejecutar una acción (act) y cómo mejorar su técnica a partir de los resultados (learn). Su estado interno (su "memoria muscular" o conocimiento) es la Q-Table o los pesos de una red neuronal.  
* El **AlgorithmPlugin** (ej: QLearningPlugin) es el **entrenador**. No corre la carrera, pero diseña y dirige todo el plan de entrenamiento (train).  
  * **build\_agent**: El entrenador "ficha" o crea a su atleta.  
  * **train**: El entrenador diseña la rutina: pone al atleta en la pista (el env), le dice cuándo correr, observa los resultados y le dice al atleta que aprenda de ellos (agent.learn).  
  * **eval**: El entrenador pone a prueba al atleta en una competición para ver su rendimiento.

#### **Responsabilidades Detalladas**

* **AlgorithmPlugin (El Orquestador)**  
  * **Gestiona el flujo:** Controla el bucle principal de episodios y pasos (for episode in episodes...).  
  * **Interactúa con el exterior:** Es el único que habla con el env (haciendo env.step()) y con el sistema de archivos (guardando y cargando en el run\_dir).  
  * **Crea el agente:** Usa build\_agent para instanciar el agente con los hiperparámetros correctos que vienen del archivo config.py.  
  * **Es genérico:** Su estructura es la misma para cualquier algoritmo (siempre tiene train y eval).  
* **Agent (El Cerebro)**  
  * **Contiene la lógica de RL:** Implementa la fórmula matemática del algoritmo (la actualización de la Q-Table, el forward pass de la red neuronal, etc.).  
  * **Toma decisiones:** Su método act o predict decide qué acción tomar dada una observación.  
  * **Aprende:** Su método learn actualiza su estado interno (la Q-Table, los pesos de la red) basándose en la experiencia (obs, action, reward, next\_obs).  
  * **Es específico:** Un QLearningAgent es muy diferente de un DQNAgent.

## **5\. Guía Práctica: Cómo Añadir un Nuevo Algoritmo (DQN)**

Siguiendo la lógica anterior, añadir un agente DQN sería un proceso muy sistemático. La clave es replicar el patrón de diseño de tu QLearningPlugin, que delega el trabajo pesado a funciones "helper".

### **Comparación de Patrones: RandomPlugin vs. QLearningPlugin**

* **RandomPlugin**: Es simple. El bucle de entrenamiento y evaluación está escrito **directamente dentro** de los métodos train y eval del plugin. Esto está bien para algoritmos sencillos.  
* **QLearningPlugin**: Es más avanzado y robusto. En lugar de tener el bucle dentro, **llama a funciones externas** (train\_with\_state\_adapter, evaluate\_with\_optional\_recording) que contienen la lógica reutilizable. **Este es el patrón que debemos seguir para DQN.**

### **Paso 1: Crear la Lógica del Agente**

Primero, necesitas el código del agente en sí. Este archivo contiene la clase con la red neuronal, el replay buffer, y los métodos act, learn, save y load.

**Nuevo Archivo: mlvlab/agents/dqn\_agent.py**

```python
# mlvlab/agents/dqn_agent.py
import torch
# ... importaciones de PyTorch ...

class DQNAgent:
    def __init__(self, observation_space, action_space, ...):
        # Lógica de la red neuronal, optimizador, replay buffer, etc.
        pass

    def act(self, obs, epsilon=0.0):
        # Lógica para elegir una acción (con epsilon-greedy)
        pass

    def learn(self):
        # Lógica para entrenar la red con un lote del replay buffer
        pass
    
    def save(self, filepath):
        # Guardar los pesos del modelo (model.state_dict())
        pass

    def load(self, filepath):
        # Cargar los pesos del modelo
        pass
```

### **Paso 2: Crear la Carpeta del Plugin**

Dentro de mlvlab/algorithms/, crea una nueva carpeta para tu algoritmo.

* mlvlab/algorithms/dqn/

### **Paso 3: Implementar el Plugin (Patrón Avanzado)**

Dentro de la nueva carpeta, crea el archivo plugin.py. Este plugin no contendrá el bucle de entrenamiento, sino que llamará a una función "helper" que sí lo hará (igual que hace el QLearningPlugin).

**Nuevo Archivo: mlvlab/algorithms/dqn/plugin.py**

```python
from __future__ import annotations
from pathlib import Path
from typing import Any, Dict, Optional
import gymnasium as gym

from mlvlab.algorithms.registry import register_algorithm
from mlvlab.agents.dqn_agent import DQNAgent # <-- Importas tu nuevo agente
# from mlvlab.helpers.train import train_with_replay_buffer # <-- Importarías un nuevo helper
# from mlvlab.helpers.eval import evaluate_with_optional_recording # <-- Reutilizas el helper de eval

class DQNPlugin:
    def key(self) -> str:
        return "dqn"

    def build_agent(self, env: gym.Env, hparams: Dict[str, Any]) -> DQNAgent:
        # Crea una instancia de tu agente DQN con los hiperparámetros
        return DQNAgent(
            env.observation_space,
            env.action_space,
            learning_rate=hparams.get("alpha", 0.001),
            # ... otros hiperparámetros del config.py ...
        )

    def train(self, env_id: str, config: Dict[str, Any], run_dir: Path, seed: Optional[int] = None, render: bool = False) -> None:
        # El plugin prepara el 'agent_builder' y delega el bucle a un helper
        def agent_builder(env: gym.Env) -> DQNAgent:
            return self.build_agent(env, config)

        # Aquí llamarías a una función helper que contenga el bucle de entrenamiento de DQN
        # train_with_replay_buffer(
        #     env_id=env_id,
        #     run_dir=run_dir,
        #     total_episodes=int(config.get("episodes", 1000)),
        #     agent_builder=agent_builder,
        #     seed=seed,
        #     render=render,
        #     # ... otros parámetros específicos de DQN (tamaño del buffer, etc.) ...
        # )
        print("Lógica de entrenamiento para DQN iría aquí, llamando a un helper.")


    def eval(self, env_id: str, run_dir: Path, **kwargs: Any) -> Optional[str]:
        # La evaluación es similar: preparas el builder y llamas al helper
        def builder(env: gym.Env) -> DQNAgent:
            agent = self.build_agent(env, {})
            # El agente DQN sabe cómo cargar sus propios pesos
            agent.load(run_dir / "model.pth") 
            return agent

        # Reutilizas el mismo helper de evaluación que Q-Learning
        # evaluate_with_optional_recording(
        #     env_id=env_id,
        #     run_dir=run_dir,
        #     agent_builder=builder,
        #     **kwargs
        # )
        print("Lógica de evaluación para DQN iría aquí, llamando a un helper.")
        return None

# ¡El paso mágico! El plugin se registra a sí mismo.
register_algorithm(DQNPlugin())
```

### **Paso 4: Actualizar la Configuración de un Entorno**

Para usar tu nuevo agente, solo tendrías que ir al config.py de cualquier entorno y cambiar el algoritmo.

**Archivo: mlvlab/envs/un\_entorno\_cualquiera/config.py**

```python
# ...
ALGORITHM = "dqn" # <-- ¡Listo!

BASELINE = {
    "config": {
        "episodes": 5000,
        "alpha": 0.001, # Learning rate para DQN
        # ... otros hiperparámetros para DQN ...
    }
}
# ...
```

¡Y ya está\! Al ejecutar mlvlab train un-entorno-cualquiera, el sistema detectaría tu nuevo plugin dqn y ejecutaría la lógica que acabas de definir.