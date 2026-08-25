<template>
	<el-config-provider :size="getGlobalComponentSize" :locale="getGlobalI18n">
		<!-- v-show="themeConfig.lockScreenTime > 1" -->
		<router-view v-show="themeConfig.lockScreenTime > 1" />
		<LockScreen v-if="themeConfig.isLockScreen" />
		<Setings ref="setingsRef" v-show="themeConfig.lockScreenTime > 1" />
		<CloseFull v-if="!themeConfig.isLockScreen" />
<!--		<Upgrade v-if="getVersion" />-->
	</el-config-provider>
</template>

<script setup lang="ts" name="app">
import { defineAsyncComponent, computed, ref, onBeforeMount, onMounted, onUnmounted, nextTick, watch, onBeforeUnmount } from 'vue';
import { useRoute, useRouter } from 'vue-router';
import { useI18n } from 'vue-i18n';
import { storeToRefs } from 'pinia';
import { useTagsViewRoutes } from '/@/stores/tagsViewRoutes';
import { useThemeConfig } from '/@/stores/themeConfig';
import other from '/@/utils/other';
import { Local, Session } from '/@/utils/storage';
import mittBus from '/@/utils/mitt';
import setIntroduction from '/@/utils/setIconfont';
import websocket from '/@/utils/websocket';

// 引入组件
const LockScreen = defineAsyncComponent(() => import('/@/layout/lockScreen/index.vue'));
const Setings = defineAsyncComponent(() => import('/@/layout/navBars/breadcrumb/setings.vue'));
const CloseFull = defineAsyncComponent(() => import('/@/layout/navBars/breadcrumb/closeFull.vue'));
const Upgrade = defineAsyncComponent(() => import('/@/layout/upgrade/index.vue'));
import { ElMessageBox, ElNotification, NotificationHandle } from 'element-plus';
import { useCore } from '/@/utils/cores';
// 定义变量内容
const { messages, locale } = useI18n();
const setingsRef = ref();
const route = useRoute();
const stores = useTagsViewRoutes();
const storesThemeConfig = useThemeConfig();
const { themeConfig } = storeToRefs(storesThemeConfig);
const core = useCore();
const router = useRouter();
// 获取版本号
const getVersion = computed(() => {
	let isVersion = false;
	if (route.path !== '/login') {
		// @ts-ignore
		if ((Local.get('version') && Local.get('version') !== __VERSION__) || !Local.get('version')) isVersion = true;
	}
	return isVersion;
});
// 获取全局组件大小
const getGlobalComponentSize = computed(() => {
	return other.globalComponentSize();
});
// 获取全局 i18n
const getGlobalI18n = computed(() => {
	return messages.value[locale.value];
});
// 设置初始化，防止刷新时恢复默认
onBeforeMount(() => {
	// 设置批量第三方 icon 图标
	setIntroduction.cssCdn();
	// 设置批量第三方 js
	setIntroduction.jsCdn();
});
// 页面加载时
onMounted(() => {
	nextTick(() => {
		// 监听布局配'置弹窗点击打开
		mittBus.on('openSetingsDrawer', () => {
			setingsRef.value.openDrawer();
		});
    // 设置皮肤缓存版本，每次更新版本可以所有用户清空缓存
		const themeConfigVersion = '1.3.0';
		// 获取缓存中的布局配置
		if (Local.get('themeConfigVersion') !== themeConfigVersion) {
			// 仅清理视觉配置，避免旧版棕色侧栏与默认品牌信息覆盖白色主题。
			Local.remove('themeConfig');
			Local.remove('themeConfigStyle');
			Local.remove('frequency');
			Local.set('themeConfigVersion', themeConfigVersion);
			window.location.reload();
			return;
		}
		if (Local.get('themeConfig')) {
			const savedConfig = Local.get('themeConfig');
			storesThemeConfig.setThemeConfig({ themeConfig: savedConfig });
			document.documentElement.style.cssText = Local.get('themeConfigStyle');
		}
		// 获取缓存中的全屏配置
		if (Session.get('isTagsViewCurrenFull')) {
			stores.setCurrenFullscreen(Session.get('isTagsViewCurrenFull'));
		}
	});
});
// 页面销毁时，关闭监听布局配置/i18n监听
onUnmounted(() => {
	mittBus.off('openSetingsDrawer', () => {});
});
// 监听路由的变化，设置网站标题
watch(
    () => route.path,
    () => {
      other.useTitle();
      other.useFavicon();
      if (!websocket.websocket) {
        //websockt 模块
        try {
          websocket.init(wsReceive)
        } catch (e) {
          console.log('websocket错误');
        }
      }
    },
    {
      deep: true,
    }
);

// websocket相关代码
import { messageCenterStore } from '/@/stores/messageCenter';
const wsReceive = (message: any) => {
  let data: any = null;
  try {
    data = JSON.parse(message.data);
  } catch (e) {
    // 非法 JSON 直接忽略，避免中断整个消息处理
    return;
  }
  const { unread } = data;
  const messageCenter = messageCenterStore();
  messageCenter.setUnread(Number.isFinite(Number(unread)) ? Number(unread) : 0);
  if (data.contentType === 'SYSTEM') {
    ElNotification({
      title: '系统消息',
      message: data.content,
      type: 'success',
      position: 'bottom-right',
      duration: 5000,
    });
  } else if (data.contentType === 'Content') {
    // 服务端内容可能含富文本，但这里以纯文本渲染，避免 HTML 注入
    ElMessageBox.confirm(String(data.content ?? ''), String(data.notificationTitle ?? '通知'), {
      confirmButtonText: String(data.notificationButton ?? '确定'),
      cancelButtonText: '关闭',
      type: 'info',
      closeOnClickModal: false,
    }).then(() => {
      ElMessageBox.close();
      const path = data.path;
      if (route.path === path) {
        core.bus.emit('onNewTask', { name: 'onNewTask' });
      } else {
        router.push({ path});
      }
    })
        .catch(() => {});
  }

};
onBeforeUnmount(() => {
  // 关闭连接
  websocket.close();
});
</script>
