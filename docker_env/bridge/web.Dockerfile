FROM node:20-alpine AS build

WORKDIR /web
COPY web/package.json web/package-lock.json ./
RUN npm ci --registry=https://registry.npmmirror.com
COPY web/ ./
RUN npm run build

FROM nginx:alpine
COPY docker_env/bridge/nginx.conf /etc/nginx/conf.d/default.conf
COPY --from=build /web/dist /usr/share/nginx/html
