from __future__ import annotations

from typing import Any, Optional, Callable
import time
from threading import Lock, Thread
import random
import atexit

from .state import StateStore


class SimulationRunner:
    """
    Ejecuta el bucle de simulación en un hilo dedicado.

    Requiere que el agente implemente al menos:
      - choose_action(state_or_obs, epsilon)
      - update(s_or_obs, action, reward, s2_or_obs, lr, gamma)
      - reset()
    """

    def __init__(self, env: Any, agent: Any, state: StateStore, env_lock: Lock, state_from_obs: Optional[Callable[[Any], Any]] = None) -> None:
        self.env = env
        self.agent = agent
        self.state = state
        self.env_lock = env_lock
        self._state_from_obs = state_from_obs

        self._thread: Optional[Thread] = None
        self._render_thread: Optional[Thread] = None
        self._stop = False
        self._last_check_time = time.time()
        self._steps_at_last_check = 0

        # Detectar si el entorno está en modo ventana humana (una sola vez)
        try:
            self._render_human = getattr(
                self.env, 'render_mode', None) == 'human'
        except Exception:
            self._render_human = False
        self._atexit_registered = False

        # Inicialización del entorno
        with self.env_lock:
            obs, info = self.env.reset()
        self.state.set(["sim", "obs"], obs)
        self.state.set(["sim", "info"], info)

    def start(self) -> None:
        if self._thread and self._thread.is_alive():
            return
        self._stop = False
        # Lanzar hilo de render secundario si aplica (30 FPS aprox)
        if getattr(self, '_render_human', False) and (not self._render_thread or not self._render_thread.is_alive()):
            self._render_thread = Thread(target=self._render_loop, daemon=True)
            self._render_thread.start()
        self._thread = Thread(target=self._loop, daemon=True)
        self._thread.start()
        # Parada automática al finalizar el proceso
        if not self._atexit_registered:
            try:
                atexit.register(self._atexit_hook)
                self._atexit_registered = True
            except Exception:
                pass

    def stop(self) -> None:
        # Señal inmediata de parada a los bucles
        self._render_human = False
        self._stop = True
        # Ocultar la ventana humana de forma inmediata si existe
        try:
            win = getattr(getattr(self.env, 'unwrapped',
                          self.env), 'window', None)
            if win is not None:
                try:
                    win.set_visible(False)  # type: ignore[attr-defined]
                except Exception:
                    try:
                        import arcade  # type: ignore
                        arcade.close_window()
                    except Exception:
                        pass
        except Exception:
            pass
        t = self._thread
        if t and t.is_alive():
            try:
                t.join(timeout=0.5)
            except Exception:
                pass
        self._thread = None
        # Esperar hilo de render y cerrar ventana humana si existe
        rt = self._render_thread
        if rt and rt.is_alive():
            try:
                rt.join(timeout=0.2)
            except Exception:
                pass
        self._render_thread = None
        try:
            if getattr(self, '_render_human', False):
                with self.env_lock:
                    self.env.close()
        except Exception:
            pass

    def _atexit_hook(self) -> None:
        # Intento de parada limpia al finalizar el script/principal
        try:
            self.stop()
        except Exception:
            pass

    def _render_loop(self) -> None:
        # Hilo independiente para mantener la ventana Arcade refrescada
        fps = 30
        try:
            fps = int(getattr(getattr(self.env, 'unwrapped', self.env),
                      'metadata', {}).get('render_fps', 30))
        except Exception:
            fps = 30
        interval = 1.0 / max(1, fps)
        # Primer render para crear ventana y desactivar cierre manual
        try:
            with self.env_lock:
                self.env.render()
                # Intentar bloquear el cierre manual de la ventana
                try:
                    win = getattr(getattr(self.env, 'unwrapped',
                                  self.env), 'window', None)
                    if win is not None:
                        try:
                            import pyglet
                            win.push_handlers(on_close=lambda: True)
                        except Exception:
                            try:
                                win.on_close = lambda *args, **kwargs: None  # type: ignore
                            except Exception:
                                pass
                except Exception:
                    pass
        except Exception:
            pass
        while not self._stop and getattr(self, '_render_human', False):
            try:
                with self.env_lock:
                    self.env.render()
            except Exception:
                pass
            # usar sleep corto para permitir salida rápida
            remaining = interval
            while remaining > 0 and not self._stop and getattr(self, '_render_human', False):
                sl = min(0.01, remaining)
                time.sleep(sl)
                remaining -= sl

    # --- interno --- #
    def _extract_state(self, obs: Any) -> Any:
        # Prioridad: callable pasado al runner > método del agente > obs sin transformar
        if callable(self._state_from_obs):
            try:
                return self._state_from_obs(obs)
            except Exception:
                pass
        extractor = getattr(self.agent, "extract_state_from_obs", None)
        if callable(extractor):
            try:
                return extractor(obs)
            except Exception:
                return obs
        return obs

    def _loop(self) -> None:
        last_step_time = time.perf_counter()
        step_accum = 0.0
        last_turbo = False

        while not self._stop:
            cmd = self.state.get(["sim", "command"]) or "run"
            if cmd == "pause":
                time.sleep(0.001)
                continue
            if cmd == "reset":
                # Generar nueva semilla para cambiar el escenario
                new_seed = random.randint(0, 1_000_000)
                with self.env_lock:
                    try:
                        obs, info = self.env.reset(seed=new_seed)
                    except TypeError:
                        # Fallback si el entorno no acepta seed
                        obs, info = self.env.reset()
                # Actualizar claves individuales sin reemplazar el dict 'sim'
                self.state.set(["sim", "obs"], obs)
                self.state.set(["sim", "info"], info)
                self.state.set(["sim", "current_episode_reward"], 0.0)
                self.state.set(["sim", "total_steps"], 0)
                self.state.set(["sim", "seed"], new_seed)
                if hasattr(self.agent, "reset"):
                    try:
                        self.agent.reset()
                    except Exception:
                        pass
                # Reset de métricas (sin reemplazar el dict)
                self.state.set(["metrics", "episodes_completed"], 0)
                self.state.set(["metrics", "reward_history"], [])
                self.state.set(["metrics", "steps_per_second"], 0)
                # Reset mínimo de hiperparámetros clave
                self.state.set(["agent", "epsilon"], 1.0)
                # Re-sincronizar acumuladores locales (no tocar turbo ni speed del estado)
                step_accum = 0.0
                last_step_time = time.perf_counter()
                # Continuar en run
                self.state.set(["sim", "command"], "run")
                continue

            now = time.perf_counter()
            dt = now - last_step_time
            # Robustez: convertir turbo a bool sin ambigüedades
            raw_turbo = self.state.get(["sim", "turbo_mode"])
            if isinstance(raw_turbo, dict) and 'value' in raw_turbo:
                raw_turbo = raw_turbo['value']
            turbo = bool(raw_turbo) if not isinstance(
                raw_turbo, str) else raw_turbo.strip().lower() in {"true", "on", "1", "yes"}
            if turbo != last_turbo:
                # Igual que en visualizer.py: al cambiar turbo, reiniciar acumulador y saltar este tick
                step_accum = 0.0
                last_step_time = now
                last_turbo = turbo
                continue
            last_step_time = now
            # sin enfriamiento adicional para respetar el comportamiento previo estable

            # Si turbo está activo, ignoramos speed_multiplier para el cálculo de steps
            spm = max(0, int(self.state.get(["sim", "speed_multiplier"]) or 1))
            if turbo:
                # Réplica de la estrategia original: muchos pasos por dt con top alto
                steps_to_do = max(
                    1, min(120000, int(30000 * dt) if dt > 0 else 10000))
            else:
                step_accum += spm * dt
                steps_to_do = int(step_accum)
            if steps_to_do <= 0:
                time.sleep(0.0005)
                continue
            steps_to_do = min(steps_to_do, 40000)

            for _ in range(steps_to_do):
                if not turbo:
                    step_accum -= 1.0
                with self.env_lock:
                    obs = self.state.get(["sim", "obs"])
                    s = self._extract_state(obs)
                    # Sincronizar hiperparámetros desde el estado al agente (si existen)
                    try:
                        if hasattr(self.agent, 'learning_rate'):
                            lr_state = self.state.get(["agent", "learning_rate"]) or getattr(
                                self.agent, 'learning_rate')
                            setattr(self.agent, 'learning_rate',
                                    float(lr_state))
                        if hasattr(self.agent, 'discount_factor'):
                            df_state = self.state.get(["agent", "discount_factor"]) or getattr(
                                self.agent, 'discount_factor')
                            setattr(self.agent, 'discount_factor',
                                    float(df_state))
                        if hasattr(self.agent, 'epsilon'):
                            eps_state = self.state.get(
                                ["agent", "epsilon"]) or getattr(self.agent, 'epsilon')
                            setattr(self.agent, 'epsilon', float(eps_state))
                    except Exception:
                        pass

                    # Elegir acción con compatibilidad (act -> choose_action)
                    if hasattr(self.agent, 'act') and callable(getattr(self.agent, 'act')):
                        action = self.agent.act(s)
                    else:
                        epsilon = float(self.state.get(
                            ["agent", "epsilon"]) or 0.0)
                        action = self.agent.choose_action(
                            s, epsilon)  # type: ignore[attr-defined]
                    next_obs, reward, terminated, truncated, info = self.env.step(
                        action)
                    # Persistir
                    self.state.set(["sim", "obs"], next_obs)
                    self.state.set(["sim", "info"], info)
                    # Señal de audio si aplica (similar al visualizer original)
                    try:
                        speed = int(self.state.get(
                            ["sim", "speed_multiplier"]) or 1)
                        if isinstance(info, dict) and ("play_sound" in info) and (not turbo) and speed <= 50:
                            self.state.set(
                                ["sim", "last_sound"], info["play_sound"])
                    except Exception:
                        pass
                s2 = self._extract_state(next_obs)
                # Aprendizaje con compatibilidad (learn -> update)
                if hasattr(self.agent, 'learn') and callable(getattr(self.agent, 'learn')):
                    try:
                        self.agent.learn(s, action, float(
                            reward), s2, bool(terminated or truncated))
                    except Exception:
                        pass
                else:
                    lr = float(self.state.get(
                        ["agent", "learning_rate"]) or 0.1)
                    gamma = float(self.state.get(
                        ["agent", "discount_factor"]) or 0.9)
                    try:
                        # type: ignore[attr-defined]
                        self.agent.update(s, action, reward, s2, lr, gamma)
                    except Exception:
                        pass

                # Métricas
                cur_rew = float(self.state.get(
                    ["sim", "current_episode_reward"]) or 0.0) + float(reward)
                self.state.set(
                    ["sim", "current_episode_reward"], round(cur_rew, 2))
                total_steps = int(self.state.get(
                    ["sim", "total_steps"]) or 0) + 1
                self.state.set(["sim", "total_steps"], total_steps)

                if terminated or truncated:
                    episodes = int(self.state.get(
                        ["metrics", "episodes_completed"]) or 0) + 1
                    self.state.set(["metrics", "episodes_completed"], episodes)
                    history = list(self.state.get(
                        ["metrics", "reward_history"]) or [])
                    history.append(round(cur_rew, 2))
                    # recorte por history_size si existe en metrics.chart_reward_number
                    max_len = int(self.state.get(
                        ["metrics", "chart_reward_number"]) or 100)
                    if len(history) > max_len:
                        history = history[-max_len:]
                    self.state.set(["metrics", "reward_history"], history)
                    with self.env_lock:
                        obs, info = self.env.reset()
                        self.state.set(["sim", "obs"], obs)
                        self.state.set(["sim", "info"], info)
                    self.state.set(["sim", "current_episode_reward"], 0.0)
                    # Si el agente gestiona epsilon internamente, reflejarlo en el estado
                    try:
                        if hasattr(self.agent, 'epsilon'):
                            self.state.set(["agent", "epsilon"], float(
                                getattr(self.agent, 'epsilon')))
                    except Exception:
                        pass
                    break

            # Steps per second
            now2 = time.time()
            elapsed = now2 - self._last_check_time
            if elapsed > 0.5:
                steps_this_interval = int(self.state.get(
                    ["sim", "total_steps"]) or 0) - self._steps_at_last_check
                sps = int(steps_this_interval / elapsed) if elapsed > 0 else 0
                self.state.set(["metrics", "steps_per_second"], sps)
                self._last_check_time = now2
                self._steps_at_last_check = int(
                    self.state.get(["sim", "total_steps"]) or 0)
            time.sleep(0.0005)
