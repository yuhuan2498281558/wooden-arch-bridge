<template>
	<div class="bridge-login-page">
		<section class="hero-panel">
			<header class="brand-row">
				<div>
					<strong>桥梁智能设计平台</strong>
					<span>编木拱廊桥参数化智能设计</span>
				</div>
			</header>

			<div class="hero-copy">
				<p>参数化设计 / 结构预测 / 安全分析</p>
				<h1>中国木拱廊桥智能设计系统</h1>
				<span>融合参数化设计、结构预测、安全性分析、三维模型展示与工程图纸生成的一体化设计工作台。</span>
			</div>

			<div class="feature-strip">
				<div>
					<strong>SSA-XGBoost</strong>
					<span>参数预测</span>
				</div>
				<div>
					<strong>CF-BPNN</strong>
					<span>智能优化</span>
				</div>
				<div>
					<strong>BIM / 3D</strong>
					<span>模型联动</span>
				</div>
			</div>

			<div class="bridge-visual">
				<img :src="bridgeLoginImage" alt="木拱廊桥三维模型" />
			</div>
		</section>

		<section class="login-panel">
			<div class="login-meta">
				<span>智能设计工作台</span>
			</div>
			<div class="login-card">
				<div class="login-card-title">
					<span>木拱廊桥设计系统</span>
					<h2>{{ userInfos.pwd_change_count === 0 ? '首次登录请修改密码' : '欢迎回来' }}</h2>
					<p>登录后进入智能设计工作台</p>
				</div>

				<div class="login-form-wrap">
					<ChangePwd v-if="userInfos.pwd_change_count === 0" />
					<Account v-else />
					<OAuth2 />
				</div>
			</div>
		</section>

		<footer class="login-footer">
			<span>{{ loginCopyright }}</span>
			<a href="https://beian.miit.gov.cn" target="_blank">{{ getSystemConfig['login.keep_record'] || '工程设计平台' }}</a>
		</footer>
	</div>
</template>

<script setup lang="ts" name="loginIndex">
import { computed, defineAsyncComponent, onMounted } from 'vue';
import { storeToRefs } from 'pinia';
import { NextLoading } from '/@/utils/loading';
import bridgeLoginImage from '/@/assets/bridge-login.png';
import { SystemConfigStore } from '/@/stores/systemConfig';
import { useUserInfo } from '/@/stores/userInfo';

const Account = defineAsyncComponent(() => import('/@/views/system/login/component/account.vue'));
const ChangePwd = defineAsyncComponent(() => import('/@/views/system/login/component/changePwd.vue'));
const OAuth2 = defineAsyncComponent(() => import('/@/views/system/login/component/oauth2.vue'));

const { userInfos } = storeToRefs(useUserInfo());
const systemConfigStore = SystemConfigStore();
const { systemConfig } = storeToRefs(systemConfigStore);

const getSystemConfig = computed(() => systemConfig.value);
const loginCopyright = computed(() => {
	const configuredText = String(getSystemConfig.value['login.copyright'] || '').trim();
	if (!configuredText || /dvadmin|django-vue/i.test(configuredText)) {
		return 'Copyright © 2026 木拱桥智能设计平台';
	}
	return configuredText;
});

onMounted(() => {
	NextLoading.done();
});
</script>

<style scoped lang="scss">
.bridge-login-page {
	position: relative;
	min-height: 100vh;
	overflow: hidden;
	background:
		linear-gradient(90deg, rgba(248, 250, 247, 0.98) 0%, rgba(246, 248, 242, 0.86) 48%, rgba(242, 245, 241, 0.98) 100%),
		#f3f5ef;
	color: #1d261c;
}

.hero-panel {
	position: absolute;
	inset: 0;
	min-width: 0;
	padding: 42px 54px 96px;
	display: flex;
	flex-direction: column;
	background:
		radial-gradient(circle at 68% 48%, rgba(199, 144, 52, 0.22), transparent 28%),
		linear-gradient(180deg, rgba(255, 255, 255, 0.66), rgba(240, 245, 239, 0.94));

	&::before {
		content: '';
		position: absolute;
		inset: 0;
		background:
			linear-gradient(90deg, rgba(255, 255, 255, 0.86), rgba(255, 255, 255, 0.22) 42%, rgba(255, 255, 255, 0.74) 82%),
			linear-gradient(180deg, rgba(255, 255, 255, 0.9), transparent 36%, rgba(248, 250, 247, 0.94));
		z-index: 1;
		pointer-events: none;
	}
}

.brand-row {
	display: flex;
	align-items: center;
	position: relative;
	z-index: 3;

	strong {
		display: block;
		font-size: 18px;
		font-weight: 700;
		color: #172033;
	}

	span {
		display: block;
		margin-top: 3px;
		font-size: 12px;
		color: #64748b;
	}
}

.hero-copy {
	position: relative;
	z-index: 3;
	margin-top: 58px;
	max-width: 780px;

	p {
		margin: 0 0 14px;
		font-size: 12px;
		font-weight: 700;
		letter-spacing: 0;
		text-transform: uppercase;
		color: #9a661e;
	}

	h1 {
		margin: 0;
		font-size: 42px;
		line-height: 1.2;
		font-weight: 800;
		letter-spacing: 0;
		color: #172033;
	}

	span {
		display: block;
		max-width: none;
		margin-top: 16px;
		font-size: 15px;
		line-height: 1.8;
		color: #53615b;
		white-space: nowrap;
	}
}

.bridge-visual {
	position: absolute;
	left: -70px;
	right: 300px;
	bottom: 54px;
	height: 64vh;
	min-height: 460px;
	display: flex;
	align-items: flex-end;
	justify-content: center;
	pointer-events: none;
	opacity: 0.98;
	z-index: 2;

	img {
		max-width: min(1280px, 112%);
		max-height: 100%;
		object-fit: contain;
		filter: drop-shadow(0 36px 42px rgba(49, 35, 15, 0.25));
	}
}

.feature-strip {
	position: relative;
	left: auto;
	right: auto;
	width: 480px;
	margin-top: 24px;
	z-index: 3;
	display: grid;
	grid-template-columns: 1.15fr 0.9fr 0.95fr;
	gap: 1px;
	border: 1px solid rgba(122, 98, 55, 0.18);
	border-radius: 8px;
	overflow: hidden;
	background: rgba(255, 255, 255, 0.58);
	backdrop-filter: blur(10px);

	div {
		padding: 10px 13px;
		background: rgba(255, 255, 255, 0.48);
	}

	strong {
		display: block;
		font-size: 14px;
		color: #273526;
	}

	span {
		display: block;
		margin-top: 5px;
		font-size: 12px;
		color: #6b7469;
	}
}

.login-panel {
	position: absolute;
	top: 0;
	right: 0;
	bottom: 0;
	width: 520px;
	z-index: 3;
	display: flex;
	align-items: center;
	justify-content: center;
	padding: 42px 54px 42px 24px;
	background:
		linear-gradient(90deg, rgba(248, 250, 247, 0), rgba(248, 250, 247, 0.72) 24%, rgba(248, 250, 247, 0.95));
}

.login-meta {
	position: absolute;
	top: 42px;
	right: 54px;
	display: inline-flex;
	align-items: center;
	gap: 10px;
	height: 34px;
	padding: 0 12px;
	border: 1px solid rgba(154, 102, 30, 0.18);
	border-radius: 6px;
	background: rgba(255, 255, 255, 0.64);
	backdrop-filter: blur(12px);
	color: #6b7469;
	font-size: 12px;

}

.login-card {
	width: 100%;
	max-width: 420px;
	padding: 38px 38px 32px;
	border: 1px solid rgba(216, 207, 188, 0.8);
	border-radius: 8px;
	background:
		linear-gradient(180deg, rgba(255, 255, 255, 0.92), rgba(251, 250, 245, 0.84));
	backdrop-filter: blur(20px);
	box-shadow: 0 30px 70px rgba(51, 37, 17, 0.2);
	transform: translateY(-24px);
}

.login-card-title {
	margin-bottom: 24px;

	span {
		font-size: 13px;
		font-weight: 700;
		color: #9a661e;
	}

	h2 {
		margin: 8px 0 8px;
		font-size: 28px;
		line-height: 1.25;
		font-weight: 800;
		color: #1d261c;
	}

	p {
		margin: 0;
		font-size: 13px;
		color: #6b7469;
		white-space: nowrap;
	}
}

.login-form-wrap {
	:deep(.el-tabs__header) {
		display: none;
	}

	:deep(.fast-title),
	:deep(.login-content-apply) {
		display: none;
	}

	:deep(.el-input__wrapper) {
		border-radius: 6px !important;
		background: rgba(255, 255, 255, 0.82);
		box-shadow: 0 0 0 1px #d8cfbc inset;
	}

	:deep(.el-input__wrapper:hover) {
		box-shadow: 0 0 0 1px #b99a62 inset;
	}

	:deep(.el-button.login-content-submit) {
		height: 42px;
		border-radius: 6px;
		background: #8f5f1f;
		border-color: #8f5f1f;
		box-shadow: 0 10px 24px rgba(143, 95, 31, 0.24);
	}
}

.login-footer {
	position: absolute;
	left: 54px;
	right: 42px;
	bottom: 10px;
	z-index: 4;
	display: flex;
	justify-content: space-between;
	gap: 16px;
	font-size: 12px;
	color: #6b7469;

	a {
		color: #6b7469;
	}
}

@media (max-width: 1180px) {
	.bridge-login-page {
		min-height: auto;
		overflow: auto;
	}

	.hero-panel {
		position: relative;
		min-height: 640px;
	}

	.login-panel {
		position: relative;
		width: auto;
		border-left: 0;
		border-top: 1px solid #dbe3ea;
		background: #f8fafc;
	}

	.login-meta {
		top: 24px;
		right: 32px;
	}

	.login-card {
		transform: none;
	}

	.bridge-visual {
		right: 0;
	}

	.feature-strip {
		right: auto;
		width: min(520px, calc(100% - 108px));
	}

	.login-footer {
		position: static;
		padding: 12px 24px;
		background: #f8fafc;
	}
}

@media (max-width: 720px) {
	.bridge-login-page {
		min-height: 100vh;
	}

	.hero-panel {
		padding: 28px 22px 96px;
		min-height: 560px;
	}

	.hero-copy {
		margin-top: 36px;

		h1 {
			font-size: 30px;
		}

		span {
			white-space: normal;
		}
	}

	.bridge-visual {
		left: 0;
		right: 0;
		bottom: 158px;
		height: 260px;
		min-height: 260px;
	}

	.feature-strip {
		left: auto;
		right: auto;
		width: auto;
		grid-template-columns: 1fr;
	}

	.login-panel {
		padding: 24px 16px;
	}

	.login-meta {
		display: none;
	}

	.login-card {
		padding: 28px 22px 24px;
	}

	.login-footer {
		flex-direction: column;
		gap: 4px;
	}
}
</style>
