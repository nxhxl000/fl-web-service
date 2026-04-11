# frontend

React-приложение веб-сервиса: лендинг, авторизация, дашборд участника,
страницы проектов с live-графиками раундов, реестр обученных моделей,
inference playground.

## Стек

- **Vite 8** — dev-сервер и сборка
- **React 19 + TypeScript**
- **Tailwind CSS 4** — через официальный Vite-плагин (`@tailwindcss/vite`),
  конфиг — пустой `@import "tailwindcss";` в `src/index.css`
- **ESLint 9** — базовая конфигурация из шаблона Vite

shadcn/ui, TanStack Query, recharts, роутинг и WebSocket-клиент будут
добавлены по мере того, как под них появятся реальные экраны.

## Команды

```bash
cd frontend

npm install    # поставить зависимости (требуется один раз)
npm run dev    # dev-сервер на http://localhost:5173 с HMR
npm run build  # типы + прод-сборка в frontend/dist/
npm run lint   # ESLint
```

## Структура

```
frontend/
  index.html          — entry point
  vite.config.ts      — Vite + React + Tailwind plugin
  tsconfig*.json      — конфиги TypeScript
  eslint.config.js    — ESLint flat config
  src/
    main.tsx          — ReactDOM root
    App.tsx           — корневой компонент (пока просто заглушка-лендинг)
    index.css         — @import "tailwindcss";
```
