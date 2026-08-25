import { RouteRecordRaw } from 'vue-router';

/**
 * 路由meta对象参数说明
 * meta: {
 *      title:          菜单栏及 tagsView 栏、菜单搜索名称（国际化）
 *      isLink：        是否超链接菜单，开启外链条件，`1、isLink: 链接地址不为空`
 *      isHide：        是否隐藏此路由
 *      isKeepAlive：   是否缓存组件状态
 *      isAffix：       是否固定在 tagsView 栏上
 *      isIframe：      是否内嵌窗口，开启条件，`1、isIframe:true 2、isLink：链接地址不为空`
 *      roles：         当前路由权限标识，取角色管理。控制路由显示、隐藏。超级管理员：admin 普通角色：common
 *      icon：          菜单、tagsView 图标，阿里：加 `iconfont xxx`，fontawesome：加 `fa xxx`
 * }
 */

/**
 * 定义动态路由
 * 前端添加路由，请在顶级节点的 `children 数组` 里添加
 * @description 未开启 isRequestRoutes 为 true 时使用（前端控制路由），开启时第一个顶级 children 的路由将被替换成接口请求回来的路由数据
 * @description 各字段请查看 `/@/views/system/menu/component/addMenu.vue 下的 ruleForm`
 * @returns 返回路由菜单数据
 */
export const dynamicRoutes: Array<RouteRecordRaw> = [
	{
		path: '/',
		name: '/',
		component: () => import('/@/layout/index.vue'),
		redirect: '/bridge/model3d',
		meta: {
			isKeepAlive: true,
		},
		children: [],
	},
	{
		path: '/personal',
		name: 'personal',
		component: () => import('/@/views/system/personal/index.vue'),
		meta: {
			title: 'message.router.personal',
			isLink: '',
			isHide: false,
			isKeepAlive: true,
			isAffix: false,
			isIframe: false,
			icon: 'iconfont icon-gerenzhongxin',
		},
	},
	// 木拱桥设计工作台：静态兜底路由。
	// 即使部署库中没有配置 bridge 菜单记录，登录后也能直接访问工作台，
	// 避免 next('/bridge/model3d') 命中 404。isHide 保证不重复出现在侧栏。
	{
		path: '/bridge/model3d',
		name: 'bridgeModel3d',
		component: () => import('/@/views/bridge/design/index.vue'),
		meta: {
			title: '木拱桥工作台',
			isLink: '',
			isHide: true,
			isKeepAlive: true,
			isAffix: false,
			isIframe: false,
			icon: 'iconfont icon-caidan',
		},
	},
	{
		path: '/bridge/drawing',
		name: 'bridgeDrawing',
		component: () => import('/@/views/bridge/design/index.vue'),
		meta: {
			title: '结构图纸',
			isLink: '',
			isHide: true,
			isKeepAlive: true,
			isAffix: false,
			isIframe: false,
		},
	},
	{
		path: '/bridge/params',
		name: 'bridgeParams',
		component: () => import('/@/views/bridge/design/index.vue'),
		meta: {
			title: '设计参数',
			isLink: '',
			isHide: true,
			isKeepAlive: true,
			isAffix: false,
			isIframe: false,
		},
	},
	{
		path: '/bridge/results',
		name: 'bridgeResults',
		component: () => import('/@/views/bridge/design/index.vue'),
		meta: {
			title: '预测结果',
			isLink: '',
			isHide: true,
			isKeepAlive: true,
			isAffix: false,
			isIframe: false,
		},
	},
	{
		path: '/bridge/safety',
		name: 'bridgeSafety',
		component: () => import('/@/views/bridge/design/index.vue'),
		meta: {
			title: '安全性分析',
			isLink: '',
			isHide: true,
			isKeepAlive: true,
			isAffix: false,
			isIframe: false,
		},
	},
	{
		path: '/bridge/optimize',
		name: 'bridgeOptimize',
		component: () => import('/@/views/bridge/design/index.vue'),
		meta: {
			title: '方案优化',
			isLink: '',
			isHide: true,
			isKeepAlive: true,
			isAffix: false,
			isIframe: false,
		},
	},
	{
		path: '/bridge/chat',
		name: 'bridgeChat',
		component: () => import('/@/views/bridge/design/index.vue'),
		meta: {
			title: '智能设计',
			isLink: '',
			isHide: true,
			isKeepAlive: true,
			isAffix: false,
			isIframe: false,
		},
	}
];

/**
 * 定义404、401界面
 * @link 参考：https://next.router.vuejs.org/zh/guide/essentials/history-mode.html#netlify
 */
export const notFoundAndNoPower = [
	{
		path: '/:path(.*)*',
		name: 'notFound',
		component: () => import('/@/views/system/error/404.vue'),
		meta: {
			title: 'message.staticRoutes.notFound',
			isHide: true,
		},
	},
	{
		path: '/401',
		name: 'noPower',
		component: () => import('/@/views/system/error/401.vue'),
		meta: {
			title: 'message.staticRoutes.noPower',
			isHide: true,
		},
	},
];

/**
 * 定义静态路由（默认路由）
 * 此路由不要动，前端添加路由的话，请在 `dynamicRoutes 数组` 中添加
 * @description 前端控制直接改 dynamicRoutes 中的路由，后端控制不需要修改，请求接口路由数据时，会覆盖 dynamicRoutes 第一个顶级 children 的内容（全屏，不包含 layout 中的路由出口）
 * @returns 返回路由菜单数据
 */
export const staticRoutes: Array<RouteRecordRaw> = [
	{
		path: '/login',
		name: 'login',
		component: () => import('/@/views/system/login/index.vue'),
		meta: {
			title: '登录',
		},
	},
	{
		path: '/demo',
		name: 'demo',
		component: () => import('/@/views/system/demo/index.vue'),
		meta: {
			title: 'message.router.personal'
		},
	}
];
