<template>
	<div class="bridge-shell" :class="{ 'workbench-mode': showChatPanel }">
		<aside v-if="showChatPanel" class="chat-panel">
			<header class="chat-header">
				<div class="bridge-mark">桥</div>
				<div>
					<h2>中国木拱廊桥智能设计系统</h2>
					<p>SSA-XGBoost + CF-BPNN</p>
				</div>
			</header>

			<div ref="messageBoxRef" class="messages">
				<div v-for="msg in messages" :key="msg.id" class="msg" :class="msg.role">
					{{ msg.text }}
				</div>
				<div v-if="chatLoading" class="msg bot loading">分析中...</div>
			</div>

			<div class="chat-input-area">
				<div class="chat-mode-selector">
					<span>五节苗外牛头节点位置</span>
					<el-radio-group v-model="form.outer_node_structure_type" size="small">
						<el-radio-button label="front_half">前半区</el-radio-button>
						<el-radio-button label="back_half">后半区</el-radio-button>
					</el-radio-group>
				</div>
				<div class="chat-row">
					<el-input
						v-model="chatText"
						type="textarea"
						:autosize="{ minRows: 3, maxRows: 5 }"
						placeholder="例如：设计一座总长18米、净跨15米、桥宽4.5米的单孔木拱廊桥"
						@keydown.enter.exact.prevent="sendMessage"
					/>
					<el-button type="primary" :icon="Position" :loading="chatLoading" @click="sendMessage" />
				</div>
			</div>
		</aside>

		<main class="result-panel">
			<el-tabs v-model="activeTab" class="bridge-tabs">
				<el-tab-pane label="3D 模型" name="model3d">
					<section class="model-stage">
						<div ref="bimfaceContainerRef" class="bimface-container"></div>

						<div v-if="!model.loaded && !model.loading && !model.error" class="model-placeholder">
							<el-icon><Box /></el-icon>
							<span>点击加载木拱廊桥 3D 模型</span>
							<el-button type="primary" :icon="View" @click="loadBimfaceModel">加载 3D 模型</el-button>
						</div>
						<div v-if="model.loading" class="model-placeholder">模型加载中...</div>
						<div v-if="model.error" class="model-placeholder error">
							<span>{{ model.error }}</span>
							<el-button @click="retryModel">重试</el-button>
						</div>

						<svg ref="labelSvgRef" class="label-svg"></svg>
						<div ref="labelLayerRef" class="label-layer"></div>

						<div v-if="model.loaded" class="model-toolbar">
							<el-button size="small" @click="clearLabels">清除标注</el-button>
							<el-button size="small" type="primary" @click="toggleLabelMode">
								标注模式：{{ labelMode ? '开' : '关' }}
							</el-button>
						</div>

						<div v-if="propertyPanel.visible" class="property-panel">
							<div class="property-title">
								<span>{{ propertyPanel.title }}</span>
								<el-button link @click="propertyPanel.visible = false">关闭</el-button>
							</div>
							<div class="property-body">
								<div v-if="!propertyPanel.rows.length" class="property-empty">暂无属性数据</div>
								<div v-for="row in propertyPanel.rows" :key="row.key" class="property-row">
									<span>{{ row.key }}</span>
									<strong>{{ row.value }}</strong>
								</div>
							</div>
						</div>
					</section>
				</el-tab-pane>

				<el-tab-pane label="结构图纸" name="drawing">
					<section class="drawing-stage">
						<div class="stage-actions">
							<el-button :icon="Picture" :loading="loading.visualize" :disabled="!lastResult" @click="runVisualize">生成图纸</el-button>
							<el-button :disabled="!imageSrc" @click="drawingZoom = Math.max(0.6, drawingZoom - 0.1)">缩小</el-button>
							<el-button :disabled="!imageSrc" @click="drawingZoom = 1">1:1</el-button>
							<el-button :disabled="!imageSrc" @click="drawingZoom = Math.min(2, drawingZoom + 0.1)">放大</el-button>
							<el-button :icon="Download" :disabled="!imageSrc" @click="downloadDrawing">下载图纸</el-button>
						</div>
						<div v-if="imageSrc" class="drawing-viewer">
							<img :src="imageSrc" alt="桥梁结构图纸" :style="{ width: `${drawingBaseWidth}px`, transform: `scale(${drawingZoom})` }" />
						</div>
						<div v-else class="empty-state">完成对话或预测后生成结构图纸</div>
					</section>
				</el-tab-pane>

				<el-tab-pane label="设计参数" name="params">
					<section class="params-stage">
						<el-form ref="formRef" :model="form" :rules="rules" label-position="top">
							<div class="params-grid">
								<el-form-item label="净跨跨度" prop="span"><el-input-number v-model="form.span" :min="3" :max="40" :step="0.1" /><span>m</span></el-form-item>
								<el-form-item label="桥面宽度" prop="width"><el-input-number v-model="form.width" :min="3" :max="8" :step="0.1" /><span>m</span></el-form-item>
								<el-form-item label="桥梁总长" prop="total_length"><el-input-number v-model="form.total_length" :min="5" :max="120" :step="0.5" /><span>m</span></el-form-item>
								<el-form-item label="孔数" prop="spans_count"><el-input-number v-model="form.spans_count" :min="1" :max="6" :step="1" /><span>孔</span></el-form-item>
								<el-form-item label="三节苗系统数" prop="n1"><el-input-number v-model="form.n1" :min="5" :max="11" :step="1" /><span>系</span></el-form-item>
								<el-form-item label="五节苗系统数" prop="n2"><el-input-number v-model="form.n2" :min="4" :max="10" :step="1" /><span>系</span></el-form-item>
							</div>
							<div class="stage-actions">
								<el-button @click="resetForm">重置</el-button>
								<el-button type="primary" :icon="TrendCharts" :loading="loading.predict" @click="runPredictFromForm">重新预测</el-button>
							</div>
						</el-form>
					</section>
				</el-tab-pane>

				<el-tab-pane label="预测结果" name="results">
					<section v-if="lastResult" class="results-stage">
						<div class="metric-grid">
							<div v-for="item in summaryCards" :key="item.label" class="metric">
								<span>{{ item.label }}</span>
								<strong>{{ item.value }}</strong>
							</div>
						</div>
						<div class="model-trace">
							<div><span>算法链路</span><strong>{{ modelTrace.method }}</strong></div>
							<div><span>参数模型版本</span><strong>{{ modelTrace.parameterVersion }}</strong></div>
							<div><span>节点来源</span><strong>{{ modelTrace.nodeSource }}</strong></div>
						</div>
						<div class="tables">
							<div>
								<h3>木构件根径推荐</h3>
								<el-table :data="diameterRows" border size="small">
									<el-table-column prop="name" label="构件类型" />
									<el-table-column prop="range" label="推荐范围" />
								</el-table>
							</div>
							<div>
								<h3>弦杆几何长度</h3>
								<el-table :data="lengthRows" border size="small">
									<el-table-column prop="name" label="构件类型" />
									<el-table-column prop="range" label="长度 / 范围" />
								</el-table>
							</div>
						</div>
						<div class="stage-actions">
							<el-button :icon="Download" @click="downloadExcel">导出 Excel</el-button>
						</div>
					</section>
					<div v-else class="empty-state">预测完成后显示结构设计结果</div>
				</el-tab-pane>

				<el-tab-pane label="安全性分析" name="safety">
					<section class="analysis-stage">
						<div class="service-life-controls">
							<label><span>木材类型</span><el-select v-model="woodType" class="wood-select">
								<el-option label="混合木材" value="default" />
								<el-option label="杉木" value="杉木" />
								<el-option label="马尾松" value="马尾松" />
								<el-option label="柏木" value="柏木" />
							</el-select></label>
							<label><span>已服役年限</span><el-input-number v-model="serviceLifeEvidence.current_age" :min="0" :max="200" :step="1" /></label>
							<label><span>环境暴露</span><el-select v-model="serviceLifeEvidence.exposure_class">
								<el-option label="有遮蔽" value="protected" />
								<el-option label="半暴露" value="sheltered" />
								<el-option label="完全暴露" value="exposed" />
							</el-select></label>
							<label><span>维护水平</span><el-select v-model="serviceLifeEvidence.maintenance_level">
								<el-option label="良好" value="good" />
								<el-option label="常规" value="regular" />
								<el-option label="较差" value="poor" />
							</el-select></label>
							<label class="treatment-control"><span>防腐处理</span><el-switch v-model="serviceLifeEvidence.preservative_treatment" /></label>
							<el-button :icon="Operation" :loading="loading.analyze" :disabled="!lastResult" @click="runAnalyze">分析安全性</el-button>
						</div>
						<div v-if="safetyResult" class="safety-report">
							<div class="score-card" :style="{ borderColor: safetyResult.overall_score?.color || '#27ae60' }">
								<div>
									<span>综合安全评分</span>
									<strong :style="{ color: safetyResult.overall_score?.color || '#27ae60' }">
										{{ safetyResult.overall_score?.score ?? '-' }}
									</strong>
								</div>
								<div>
									<span>等级</span>
									<strong>{{ safetyResult.overall_score?.grade || '-' }} / {{ safetyResult.overall_score?.label || '-' }}</strong>
								</div>
								<div>
									<span>最大构件 DCR</span>
									<strong>{{ safetyResult.overall_score?.system_dcr ?? safetyResult.structural_check?.max_dcr ?? '-' }}</strong>
								</div>
								<div>
									<span>预计寿命</span>
									<strong>{{ safetyResult.service_life?.estimated_life ?? '-' }} 年</strong>
								</div>
							</div>
							<div class="score-breakdown">
								<div>
									<span>结构验算</span>
									<strong>{{ formatNumber(safetyResult.overall_score?.breakdown?.reliability, 1) }}</strong>
								</div>
								<div>
									<span>耐久 / 截面寿命</span>
									<strong>{{ formatNumber(safetyResult.overall_score?.breakdown?.decay, 1) }}</strong>
								</div>
								<div>
									<span>寿命区间</span>
									<strong>{{ formatNumber(safetyResult.overall_score?.breakdown?.life, 1) }}</strong>
								</div>
								<div>
									<span>最大 DCR</span>
									<strong>{{ safetyResult.member_failure?.system_dcr ?? '-' }}</strong>
								</div>
							</div>

							<div v-if="safetyResult.integrated_assessment" class="report-note integration-note">
								<strong>安全分析已联动：</strong>
								结构包络验算 → 截面退化寿命 → 贝叶斯寿命区间 → 综合结论。
								控制寿命 <b>{{ safetyResult.integrated_assessment.governing_life_years }} 年</b>，
								来源：{{ safetyResult.integrated_assessment.governing_limit_state }}。
								综合评分权重：结构 DCR 55%，截面寿命 25%，贝叶斯寿命 20%。
							</div>


							<div class="report-section">
								<div class="section-title">
									<h3>构件可靠性分析</h3>
									<span>{{ safetyResult.member_failure?.method_note }}</span>
								</div>
								<el-table :data="safetyResult.member_failure?.members || []" border size="small">
									<el-table-column prop="name" label="构件" min-width="110" />
									<el-table-column prop="load_type" label="受力" width="90" />
									<el-table-column prop="section_area" label="截面积 cm²" width="110" />
									<el-table-column prop="d_range" label="根径" width="120" />
									<el-table-column prop="axial_force" label="轴力 kN" width="100" />
									<el-table-column prop="dcr" label="DCR" width="90" />
									<el-table-column prop="governing" label="控制项" width="110" />
									<el-table-column label="等级" width="110">
										<template #default="{ row }">
											<el-tag :color="row.grade_color" effect="dark">{{ row.grade }}</el-tag>
										</template>
									</el-table-column>
								</el-table>
								<div class="report-note">
									<strong>建议：</strong>{{ safetyResult.member_failure?.recommendation || '暂无专项建议' }}
								</div>
							</div>

							<div class="report-section">
								<div class="section-title">
									<h3>三节苗 / 五节苗承重验算</h3>
									<span>{{ safetyResult.structural_check?.method }} · {{ safetyResult.structural_check?.model_version }}</span>
								</div>
								<div v-if="safetyResult.structural_check" class="structural-summary">
									<div :class="safetyResult.structural_check.passes ? 'ok' : 'bad'">
										<span>验算结论</span>
										<strong>{{ safetyResult.structural_check.passes ? '满足包络验算' : '不满足包络验算' }}</strong>
									</div>
									<div>
										<span>最大需求/能力比 DCR</span>
										<strong>{{ safetyResult.structural_check.max_dcr }}</strong>
									</div>
									<div>
										<span>能力/设计荷载倍数</span>
										<strong>{{ safetyResult.structural_check.capacity_load_factor ?? '-' }}</strong>
									</div>
									<div>
										<span>总设计荷载</span>
										<strong>{{ safetyResult.structural_check.load_model?.total_design_load_kn }} kN</strong>
									</div>
								</div>
								<el-table :data="safetyResult.structural_check.members || []" border size="small" class="structural-table">
									<el-table-column prop="name" label="构件" min-width="110" />
									<el-table-column prop="system" label="体系" width="110">
										<template #default="{ row }">{{ row.system === 'three_miao' ? '三节苗' : '五节苗' }}</template>
									</el-table-column>
									<el-table-column prop="length_m" label="长度 m" width="90" />
									<el-table-column prop="diameter_mm" label="代表直径 mm" width="110" />
									<el-table-column prop="axial_demand_kn" label="轴力需求 kN" width="110" />
									<el-table-column prop="stress_dcr" label="应力 DCR" width="100" />
									<el-table-column prop="buckling_dcr" label="屈曲 DCR" width="100" />
									<el-table-column prop="dcr" label="综合 DCR" width="100" />
									<el-table-column label="结论" width="110">
										<template #default="{ row }">
											<el-tag :type="row.passes ? 'success' : 'danger'">{{ row.passes ? '通过' : '不通过' }}</el-tag>
										</template>
									</el-table-column>
								</el-table>
								<div class="report-note">
									<strong>荷载与材料假设：</strong>恒载 {{ safetyResult.structural_check.assumptions?.dead_load_kpa }} kPa，活载 {{ safetyResult.structural_check.assumptions?.live_load_kpa }} kPa，组合系数 {{ safetyResult.structural_check.assumptions?.load_combination_factor }}；顺纹抗压 {{ safetyResult.structural_check.assumptions?.compressive_strength_mpa }} MPa，弹性模量 {{ safetyResult.structural_check.assumptions?.elastic_modulus_mpa }} MPa。以上为待确认方案阶段值。
								</div>
								<div class="report-note">
									<strong>截面寿命筛选：</strong>按“年径向腐朽深度”假设计算 DCR 达到 1.0 的年限；不是实际案例寿命。
								</div>
								<el-table :data="structuralDecayRows" border size="small" class="structural-table">
									<el-table-column prop="label" label="暴露情景" min-width="120" />
									<el-table-column prop="radial_loss_mm_per_year" label="年腐朽深度 mm" width="140" />
									<el-table-column prop="critical_member" label="临界构件" min-width="120" />
									<el-table-column prop="years_until_dcr_1" label="DCR=1 年限" width="120" />
								</el-table>
								<ul v-if="safetyResult.structural_check.limitations?.length" class="report-note">
									<li v-for="item in safetyResult.structural_check.limitations" :key="item">{{ item }}</li>
								</ul>
							</div>

							<div class="report-grid">
								<div class="report-section">
									<div class="section-title">
										<h3>腐蚀 / 腐朽风险评估</h3>
										<span>总体风险：{{ safetyResult.decay_risk?.overall_risk || '-' }}，巡检周期：{{ safetyResult.decay_risk?.inspect_cycle || '-' }}</span>
									</div>
									<div class="risk-list">
										<div v-for="item in safetyResult.decay_risk?.members_risk || []" :key="item.name" class="risk-item">
											<div class="risk-content">
												<div class="risk-head">
													<strong>{{ item.name }}</strong>
													<el-tag :color="item.color" effect="dark">{{ item.risk_level }}</el-tag>
												</div>
												<ul v-if="item.reasons?.length">
													<li v-for="reason in item.reasons" :key="reason">{{ reason }}</li>
												</ul>
												<span><b>防护：</b>{{ item.protection }}</span>
											</div>
										</div>
									</div>
									<div class="report-note">
										<strong>易腐蚀部位：</strong>{{ (safetyResult.decay_risk?.vulnerable_parts || []).join('、') || '暂无明显薄弱部位' }}
									</div>
								</div>

								<div class="report-section">
									<div class="section-title">
										<h3>使用年限预测</h3>
										<span>{{ safetyResult.service_life?.method }}</span>
									</div>
									<div class="life-summary">
										<div>木材：{{ safetyResult.service_life?.wood_label }}</div>
										<div>先验寿命：{{ safetyResult.service_life?.prior_life }} 年</div>
										<div>预计总寿命：{{ safetyResult.service_life?.estimated_life }} 年</div>
										<div>预计剩余寿命：{{ safetyResult.service_life?.remaining_life }} 年</div>
										<div>控制寿命：{{ safetyResult.integrated_assessment?.governing_life_years ?? '-' }} 年</div>
										<div>控制依据：{{ safetyResult.integrated_assessment?.governing_limit_state ?? '-' }}</div>

										<div>{{ safetyResult.service_life?.interval_label || '90% 预测区间' }}：{{ safetyResult.service_life?.ci_90?.[0] }} - {{ safetyResult.service_life?.ci_90?.[1] }} 年</div>
										<div>未来 10 年存续概率：{{ formatPercent(safetyResult.service_life?.survival_probability_next_10y) }}</div>
									</div>
									<div v-if="safetyResult.service_life?.confidence" class="confidence-note">
										{{ safetyResult.service_life.confidence }}
									</div>
									<el-timeline>
										<el-timeline-item
											v-for="phase in safetyResult.service_life?.phases || []"
											:key="phase.phase"
											:color="phase.color"
											:timestamp="phase.range"
										>
											<strong>{{ phase.phase }}：{{ phase.status }}</strong>
											<div>{{ phase.action }}</div>
										</el-timeline-item>
									</el-timeline>
								</div>
							</div>
						</div>
						<div v-else class="empty-state">生成设计后查看安全性分析</div>
					</section>
				</el-tab-pane>

				<el-tab-pane label="方案优化" name="optimize">
					<section class="analysis-stage">
						<div class="optimize-form">
							<el-form :model="form" label-position="top">
								<div class="optimize-inputs">
									<el-form-item label="净跨跨度"><el-input-number v-model="form.span" :min="3" :max="40" :step="0.1" /><span>m</span></el-form-item>
									<el-form-item label="桥面宽度"><el-input-number v-model="form.width" :min="3" :max="8" :step="0.1" /><span>m</span></el-form-item>
									<el-form-item label="孔数"><el-input-number v-model="form.spans_count" :min="1" :max="6" :step="1" /><span>孔</span></el-form-item>
									<el-form-item label="桥梁总长"><el-input-number v-model="form.total_length" :min="5" :max="120" :step="0.5" /><span>m</span></el-form-item>
								</div>
							</el-form>
							<el-button type="primary" :icon="Aim" :loading="loading.optimize" @click="runOptimize">生成优化方案</el-button>
						</div>
						<div v-if="optimizeRows.length" class="optimize-summary">
							<span>Pareto 候选方案 {{ optimizeRows.length }} 个</span>
							<span>以原方案为基准，绿色为提升，红色为下降</span>
						</div>
						<div v-if="optimizeOriginal" class="original-plan-panel">
							<div class="original-title">
								<div>
									<h3>原方案</h3>
									<span>当前输入参数归一化后的基准方案</span>
								</div>
								<strong>综合评分 {{ formatNumber(optimizeOriginal.score?.composite, 2) }}</strong>
							</div>
							<div class="original-grid">
								<div><span>三节苗 / 五节苗</span><strong>{{ optimizeOriginal.params?.n1 }} / {{ optimizeOriginal.params?.n2 }} 系</strong></div>
								<div><span>矢高</span><strong>{{ formatNumber(optimizeOriginal.result?.rise_height, 3) }} m</strong></div>
								<div><span>平均根径</span><strong>{{ formatNumber(optimizeOriginal.avg_diam, 1) }} mm</strong></div>
								<div><span>安全性</span><strong>{{ formatNumber(optimizeOriginal.score?.safety, 1) }}</strong></div>
								<div><span>经济性</span><strong>{{ formatNumber(optimizeOriginal.score?.economy, 1) }}</strong></div>
								<div><span>拱形质量</span><strong>{{ formatNumber(optimizeOriginal.score?.arch_quality, 1) }}</strong></div>
								<div><span>冗余度</span><strong>{{ formatNumber(optimizeOriginal.score?.redundancy, 1) }}</strong></div>
							</div>
						</div>
						<div v-if="optimizeRows.length" class="optimize-grid">
							<div v-for="item in optimizeRows" :key="item.rank" class="plan-card">
								<div class="plan-head" :style="{ borderColor: rankColor(item.rank) }">
									<div>
										<h3>{{ item.rank_label || `方案 ${item.rank}` }}</h3>
										<span>综合评分 {{ formatNumber(item.score?.composite, 2) }}</span>
									</div>
									<span class="delta-pill" :class="deltaClass(item.delta?.composite)">
										{{ deltaText(item.delta?.composite, 2) }}
									</span>
									<el-tag v-if="item.is_pareto" type="success">Pareto</el-tag>
								</div>
								<div class="plan-metrics">
									<div>
										<span>安全性</span>
										<strong>{{ formatNumber(item.score?.safety, 1) }}</strong>
										<em :class="deltaClass(item.delta?.safety)">{{ deltaText(item.delta?.safety, 1) }}</em>
									</div>
									<div>
										<span>经济性</span>
										<strong>{{ formatNumber(item.score?.economy, 1) }}</strong>
										<em :class="deltaClass(item.delta?.economy)">{{ deltaText(item.delta?.economy, 1) }}</em>
									</div>
									<div>
										<span>拱形质量</span>
										<strong>{{ formatNumber(item.score?.arch_quality, 1) }}</strong>
										<em :class="deltaClass(item.delta?.arch_quality)">{{ deltaText(item.delta?.arch_quality, 1) }}</em>
									</div>
									<div>
										<span>冗余度</span>
										<strong>{{ formatNumber(item.score?.redundancy, 1) }}</strong>
										<em :class="deltaClass(item.delta?.redundancy)">{{ deltaText(item.delta?.redundancy, 1) }}</em>
									</div>
								</div>
								<div class="plan-detail">
									<div><span>三节苗 / 五节苗</span><strong>{{ item.params?.n1 }} / {{ item.params?.n2 }} 系</strong><em :class="deltaClass(item.delta?.n1)">{{ deltaText(item.delta?.n1, 0) }} / {{ deltaText(item.delta?.n2, 0) }}</em></div>
									<div><span>矢高</span><strong>{{ formatNumber(item.result?.rise_height, 3) }} m</strong><em :class="deltaClass(item.delta?.rise_height)">{{ deltaText(item.delta?.rise_height, 3) }} m</em></div>
									<div><span>平均根径</span><strong>{{ formatNumber(item.avg_diam, 1) }} mm</strong><em :class="deltaCostClass(item.delta?.avg_diam)">{{ deltaText(item.delta?.avg_diam, 1) }} mm</em></div>
									<div><span>三节苗平弦</span><strong>{{ range(item.result?.s1_flat_lo, item.result?.s1_flat_hi, ' mm') }}</strong></div>
								</div>
								<el-button type="primary" plain @click="applyOptimizedPlan(item)">应用此方案</el-button>
							</div>
						</div>
						<div v-else class="empty-state">点击按钮生成可选优化方案</div>
					</section>
				</el-tab-pane>
			</el-tabs>
		</main>
	</div>
</template>

<script setup lang="ts">
import { computed, nextTick, onMounted, reactive, ref, watch } from 'vue';
import { useRoute, useRouter } from 'vue-router';
import { ElMessage, type FormInstance, type FormRules } from 'element-plus';
import { Aim, Box, Download, Operation, Picture, Position, TrendCharts, View } from '@element-plus/icons-vue';
import {
	analyzeBridge,
	chatBridge,
	exportBridgeExcel,
	getBimfaceComponentProperties,
	getBimfaceViewToken,
	optimizeBridge,
	predictBridge,
	visualizeBridge,
	type BridgeParams,
	type ServiceLifeEvidence,
} from './api';

declare const BimfaceSDKLoaderConfig: any;
declare const BimfaceSDKLoader: any;
declare const Glodon: any;

const activeTab = ref('model3d');
const route = useRoute();
const router = useRouter();
const formRef = ref<FormInstance>();
const messageBoxRef = ref<HTMLElement>();
const bimfaceContainerRef = ref<HTMLElement>();
const labelSvgRef = ref<SVGElement>();
const labelLayerRef = ref<HTMLElement>();

const chatText = ref('');
const chatLoading = ref(false);
const history = ref<any[]>([]);
const messages = ref([
	{
		id: 1,
		role: 'bot',
		text: '您好，我是木拱廊桥智能设计助手。选择牛头节点位置后，请描述净跨、桥宽等设计需求。',
	},
]);

const form = reactive<BridgeParams>({
	total_length: 18,
	width: 4.5,
	span: 15,
	spans_count: 1,
	n1: 9,
	n2: 8,
	rise: null,
	outer_node_structure_type: 'back_half',
});
const lastParams = ref<BridgeParams | null>(null);
const lastResult = ref<any>(null);
const imageSrc = ref('');
const drawingZoom = ref(1);
const drawingBaseWidth = ref(2200);
const drawingVersion = ref(0);
const safetyResult = ref<any>(null);
const bridgeImageKey = `bridge-design-workbench-image-v${bridgeStateSchemaVersion}`;

const structuralDecayRows = computed(() => {
	const proxy = safetyResult.value?.structural_check?.decay_life_proxy;
	if (!proxy) return [];
	return Object.entries(proxy).map(([key, value]: [string, any]) => ({ key, ...value }));
});

const optimizeOriginal = ref<any>(null);
const optimizeRows = ref<any[]>([]);
const woodType = ref('default');
const serviceLifeEvidence = reactive<ServiceLifeEvidence>({
	current_age: 0,
	exposure_class: 'sheltered',
	maintenance_level: 'regular',
	preservative_treatment: true,
});
const loading = reactive({ predict: false, visualize: false, analyze: false, optimize: false });
const bridgeStateSchemaVersion = 3;
const bridgeStateKey = `bridge-design-workbench-state-v${bridgeStateSchemaVersion}`;

const model = reactive({ loaded: false, loading: false, error: '' });
const labelMode = ref(true);
const propertyPanel = reactive<{ visible: boolean; title: string; rows: Array<{ key: string; value: string }> }>({
	visible: false,
	title: '构件属性',
	rows: [],
});
let labelCounter = 0;
let lastClick = { x: 0, y: 0 };
const labelColors = ['#1565C0', '#C62828', '#2E7D32', '#6A1B9A', '#E65100', '#00838F'];

const rules: FormRules = {
	total_length: [{ required: true, message: '请填写桥梁总长', trigger: 'blur' }],
	width: [{ required: true, message: '请填写桥面宽度', trigger: 'blur' }],
	span: [{ required: true, message: '请填写净跨跨度', trigger: 'blur' }],
	spans_count: [{ required: true, message: '请填写孔数', trigger: 'blur' }],
	n1: [{ required: true, message: '请填写三节苗系统数', trigger: 'blur' }],
	n2: [{ required: true, message: '请填写五节苗系统数', trigger: 'blur' }],
};

const fmt = (value: any, unit = '') => (value === undefined || value === null ? '-' : `${value}${unit}`);
const formatPercent = (value: any) => (Number.isFinite(Number(value)) ? `${(Number(value) * 100).toFixed(1)}%` : '-');
const range = (lo: any, hi: any, unit = '') => `${fmt(lo)} - ${fmt(hi, unit)}`;
const formatNumber = (value: any, digits = 2) => {
	if (value === undefined || value === null || value === '') return '-';
	const num = Number(value);
	return Number.isFinite(num) ? num.toFixed(digits).replace(/\.0+$/, '').replace(/(\.\d*?)0+$/, '$1') : value;
};
const deltaClass = (value: any) => {
	const num = Number(value);
	if (!Number.isFinite(num) || Math.abs(num) < 0.0001) return 'same';
	return num > 0 ? 'up' : 'down';
};
const deltaCostClass = (value: any) => {
	const num = Number(value);
	if (!Number.isFinite(num) || Math.abs(num) < 0.0001) return 'same';
	return num < 0 ? 'up' : 'down';
};
const deltaText = (value: any, digits = 1) => {
	const num = Number(value);
	if (!Number.isFinite(num) || Math.abs(num) < 0.0001) return '持平';
	const text = formatNumber(Math.abs(num), digits);
	return `${num > 0 ? '+' : '-'}${text}`;
};
const rankColor = (rank: number) => (rank === 1 ? '#27ae60' : rank === 2 ? '#2980b9' : '#8e44ad');
const showChatPanel = computed(() => activeTab.value === 'model3d');

const summaryCards = computed(() => [
	{ label: '矢跨比', value: fmt(lastResult.value?.rise_span) },
	{ label: '矢高', value: fmt(lastResult.value?.rise_height, ' m') },
	{ label: '外节点比例 α', value: fmt(lastResult.value?.geometry?.five_miao_bullhead?.ratios?.alpha_outer) },
	{ label: '内节点比例 β', value: fmt(lastResult.value?.geometry?.five_miao_bullhead?.ratios?.beta_inner) },
	{
		label: '外节点结构',
		value: ({ front_half: '前半区', back_half: '后半区' } as Record<string, string>)[
			lastResult.value?.geometry?.five_miao_bullhead?.design_mode?.requested
		] || 'v6 未分型',
	},
	{ label: '可信度', value: fmt(lastResult.value?.validation?.trust_score, ' 分') },
	{ label: '可信等级', value: lastResult.value?.validation?.trust_level || '-' },
]);
const modelTrace = computed(() => {
	const source = lastResult.value?.geometry?.five_miao_bullhead?.source;
	return {
		method: lastResult.value?.validation?.method || '-',
		parameterVersion: lastResult.value?.validation?.model_version || '规则回退',
		nodeSource:
			source === 'designer_mode_pilot'
				? '设计人员选型双专家 Pilot'
				: source === 'pilot_model'
					? '五节苗节点 v6 Pilot'
					: '固定比例回退',
	};
});
const diameterRows = computed(() =>
	lastResult.value
		? [
				{ name: '三节苗平弦', range: range(lastResult.value.s1_flat_lo, lastResult.value.s1_flat_hi, ' mm') },
				{ name: '三节苗斜弦', range: range(lastResult.value.s1_diag_lo, lastResult.value.s1_diag_hi, ' mm') },
				{ name: '五节苗平弦', range: range(lastResult.value.s2_flat_lo, lastResult.value.s2_flat_hi, ' mm') },
				{ name: '五节苗斜弦 1', range: range(lastResult.value.s2_diag1_lo, lastResult.value.s2_diag1_hi, ' mm') },
				{ name: '五节苗斜弦 2', range: range(lastResult.value.s2_diag2_lo, lastResult.value.s2_diag2_hi, ' mm') },
		  ]
		: []
);
const lengthRows = computed(() =>
	lastResult.value
		? [
				{ name: '三节苗平弦', range: fmt(lastResult.value.s1_flat_len, ' m') },
				{ name: '三节苗斜弦', range: fmt(lastResult.value.s1_diag_len, ' m') },
				{ name: '五节苗斜弦 1', range: fmt(lastResult.value.s2_diag1_len, ' m') },
				{ name: '五节苗平弦', range: fmt(lastResult.value.s2_flat_len, ' m') },
				{ name: '五节苗斜弦 2', range: fmt(lastResult.value.s2_diag2_len, ' m') },
		  ]
		: []
);

const currentParams = (): BridgeParams => ({
	...form,
	rise: form.rise || null,
	outer_node_structure_type: form.outer_node_structure_type || 'back_half',
});
const paramsSignature = (params: BridgeParams | null) => params
	? JSON.stringify([
		Number(params.total_length), Number(params.width), Number(params.span), Number(params.spans_count),
		Number(params.n1), Number(params.n2), params.rise ? Number(params.rise) : null,
		params.outer_node_structure_type || null,
	])
	: '';
const predictionIsStale = computed(() => Boolean(
	lastResult.value && lastParams.value && paramsSignature(currentParams()) !== paramsSignature(lastParams.value)
));
const freshPredictionParams = (): BridgeParams | null => {
	if (!lastResult.value || !lastParams.value) return null;
	if (predictionIsStale.value) {
		ElMessage.warning('设计参数已变更，请重新预测后再生成图纸、分析或导出');
		void goToTab('params');
		return null;
	}
	return { ...lastParams.value };
};
const scrollMessages = () => nextTick(() => messageBoxRef.value && (messageBoxRef.value.scrollTop = messageBoxRef.value.scrollHeight));
const saveBridgeState = () => {
	sessionStorage.setItem(
		bridgeStateKey,
		JSON.stringify({
			schemaVersion: bridgeStateSchemaVersion,
			form: currentParams(),
			lastParams: lastParams.value,
			lastResult: lastResult.value,
			drawingVersion: drawingVersion.value,
			safetyResult: safetyResult.value,
			optimizeOriginal: optimizeOriginal.value,
			optimizeRows: optimizeRows.value,
			history: history.value,
			messages: messages.value,
			woodType: woodType.value,
			serviceLifeEvidence: { ...serviceLifeEvidence },
		})
	);
};
const saveBridgeImage = () => {
	if (imageSrc.value) sessionStorage.setItem(bridgeImageKey, imageSrc.value);
	else sessionStorage.removeItem(bridgeImageKey);
};
const clearBridgeImage = () => {
	imageSrc.value = '';
	sessionStorage.removeItem(bridgeImageKey);
};
const restoreBridgeState = () => {
	try {
		const raw = sessionStorage.getItem(bridgeStateKey);
		if (!raw) return;
		const state = JSON.parse(raw);
		if (state.schemaVersion !== bridgeStateSchemaVersion) {
			sessionStorage.removeItem(bridgeStateKey);
			return;
		}
		if (state.form) Object.assign(form, state.form);
		lastParams.value = state.lastParams || null;
		lastResult.value = state.lastResult || null;
		const restoredSnapshot = lastResult.value?.design_inputs;
		const restoredMode = lastResult.value?.geometry?.five_miao_bullhead?.design_mode;
		const requestedMode = lastParams.value?.outer_node_structure_type;
		const snapshotMatches = Boolean(
			restoredSnapshot && lastParams.value && paramsSignature(restoredSnapshot) === paramsSignature(lastParams.value)
		);
		const modeMatches = !requestedMode || restoredMode?.requested === requestedMode;
		if (lastResult.value && (!snapshotMatches || !modeMatches)) {
			lastParams.value = null;
			lastResult.value = null;
		}
		imageSrc.value = sessionStorage.getItem(bridgeImageKey) || '';
		drawingVersion.value = state.drawingVersion || 0;
		safetyResult.value = state.safetyResult || null;
		optimizeOriginal.value = state.optimizeOriginal || null;
		optimizeRows.value = state.optimizeRows || [];
		history.value = state.history || [];
		if (state.messages?.length) messages.value = state.messages;
		woodType.value = state.woodType || 'default';
		if (state.serviceLifeEvidence) Object.assign(serviceLifeEvidence, state.serviceLifeEvidence);
	} catch {
		sessionStorage.removeItem(bridgeStateKey);
	}
};

const addMessage = (role: 'user' | 'bot' | 'system', text: string) => {
	messages.value.push({ id: Date.now() + Math.random(), role, text });
	saveBridgeState();
	scrollMessages();
};

const applyParams = (params: any) => {
	if (!params) return;
	Object.assign(form, {
		span: Number(params.span ?? form.span),
		width: Number(params.width ?? form.width),
		total_length: Number(params.total_length ?? params.length ?? form.total_length),
		spans_count: Number(params.spans_count ?? form.spans_count),
		n1: Number(params.n1 ?? form.n1),
		n2: Number(params.n2 ?? form.n2),
		rise: params.rise ?? form.rise,
		outer_node_structure_type: params.outer_node_structure_type ?? form.outer_node_structure_type,
	});
};

const goToTab = async (tab: string) => {
	const targetPath = tabPathMap[tab];
	if (!targetPath) {
		activeTab.value = tab;
		return;
	}
	if (route.path !== targetPath) {
		const query = Object.fromEntries(Object.entries(route.query).filter(([key]) => key !== 'tab'));
		await router.replace({ path: targetPath, query });
	}
	activeTab.value = tab;
};


const sendMessage = async () => {
	const text = chatText.value.trim();
	if (!text || chatLoading.value) return;
	chatText.value = '';
	addMessage('user', text);
	chatLoading.value = true;
	try {
		const { data } = await chatBridge({
			message: text,
			history: history.value,
			outer_node_structure_type: form.outer_node_structure_type || 'back_half',
		});
		if (data.status === 'need_params') {
			addMessage('bot', normalizeBotText(data.message || data.question || '还需要补充设计参数。'));
			history.value.push({ role: 'user', content: text });
			history.value.push({ role: 'assistant', content: data.message, params: data.params, missing: data.missing });
			applyParams(data.params);
		} else if (data.status === 'ready') {
			addMessage('bot', normalizeBotText(data.message || '设计完成，已生成预测结果。'));
			history.value.push({ role: 'user', content: text });
			history.value.push({ role: 'assistant', content: data.message, params: data.params });
			applyParams(data.params);
			lastParams.value = currentParams();
			lastResult.value = data.result;
			imageSrc.value = data.image ? `data:image/png;base64,${data.image}` : '';
			saveBridgeImage();
			optimizeOriginal.value = null;
			optimizeRows.value = [];
			saveBridgeState();
			await goToTab('drawing');
		} else {
			addMessage('bot', normalizeBotText(data.message || JSON.stringify(data)));
		}
	} catch (error: any) {
		addMessage('bot', error?.response?.data?.detail || '连接桥梁算法服务失败，请确认 8001 服务已启动。');
	} finally {
		chatLoading.value = false;
	}
};

const normalizeBotText = (text: string) => text
	.replaceAll('净跨度', '净跨跨度')
	.replaceAll('桥宽', '桥面宽度');

const runPredictFromForm = async () => {
	await formRef.value?.validate();
	loading.predict = true;
	try {
		const { data } = await predictBridge(currentParams());
		applyParams(data.params);
		lastParams.value = currentParams();
		lastResult.value = data.result;
		clearBridgeImage();
		safetyResult.value = null;
		optimizeOriginal.value = null;
		optimizeRows.value = [];
		saveBridgeState();
		await goToTab('results');
		ElMessage.success('预测完成');
	} catch (error: any) {
		ElMessage.error(error?.response?.data?.detail || '预测失败');
	} finally {
		loading.predict = false;
	}
};

const runVisualize = async () => {
	const predictionParams = freshPredictionParams();
	if (!predictionParams) return;
	loading.visualize = true;
	try {
		const { data } = await visualizeBridge(predictionParams, lastResult.value);
		imageSrc.value = `data:image/${data.fmt || 'png'};base64,${data.image}`;
		saveBridgeImage();
		drawingZoom.value = 1;
		drawingVersion.value = Date.now();
		saveBridgeState();
		await goToTab('drawing');
	} catch (error: any) {
		ElMessage.error(error?.response?.data?.detail || '图纸生成失败');
	} finally {
		loading.visualize = false;
	}
};

const runAnalyze = async () => {
	const predictionParams = freshPredictionParams();
	if (!predictionParams) return;
	loading.analyze = true;
	try {
		const { data } = await analyzeBridge(predictionParams, lastResult.value, woodType.value, { ...serviceLifeEvidence });
		safetyResult.value = data.result;
		saveBridgeState();
		await goToTab('safety');
	} catch (error: any) {
		ElMessage.error(error?.response?.data?.detail || '安全分析失败');
	} finally {
		loading.analyze = false;
	}
};

const runOptimize = async () => {
	loading.optimize = true;
	try {
		const { data } = await optimizeBridge({
			span: form.span,
			width: form.width,
			spans_count: form.spans_count,
			total_length: form.total_length,
			n1: form.n1,
			n2: form.n2,
			rise: form.rise,
			outer_node_structure_type: form.outer_node_structure_type || 'back_half',
		});
		optimizeOriginal.value = data.original_plan || null;
		optimizeRows.value = data.candidates || [];
		saveBridgeState();
		await goToTab('optimize');
	} catch (error: any) {
		ElMessage.error(error?.response?.data?.detail || '方案优化失败');
	} finally {
		loading.optimize = false;
	}
};

const downloadExcel = async () => {
	const predictionParams = freshPredictionParams();
	if (!predictionParams) return;
	try {
		const response = await exportBridgeExcel(predictionParams, lastResult.value);
		const blob = new Blob([response.data], { type: 'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet' });
		const link = document.createElement('a');
		link.href = URL.createObjectURL(blob);
		link.download = `桥梁设计结果_${Date.now()}.xlsx`;
		link.click();
		URL.revokeObjectURL(link.href);
	} catch (error: any) {
		// 接口失败时给出提示，避免把 JSON 错误体当 blob 下载成损坏的 xlsx
		ElMessage.error(error?.response?.data?.detail || '导出 Excel 失败，请稍后重试');
	}
};

const downloadDrawing = () => {
	if (!imageSrc.value) return;
	const link = document.createElement('a');
	link.href = imageSrc.value;
	link.download = `桥梁结构设计图_${Date.now()}.png`;
	link.click();
};

const resetForm = () => {
	Object.assign(form, {
		total_length: 18,
		width: 4.5,
		span: 15,
		spans_count: 1,
		n1: 9,
		n2: 8,
		rise: null,
		outer_node_structure_type: 'back_half',
	});
	lastParams.value = null;
	lastResult.value = null;
	clearBridgeImage();
	safetyResult.value = null;
	optimizeOriginal.value = null;
	optimizeRows.value = [];
	saveBridgeState();
	void goToTab('params');
};

const applyOptimizedPlan = async (item: any) => {
	if (!item?.params || !item?.result) return;
	Object.assign(form, {
		total_length: item.params.total_length,
		width: item.params.width,
		span: item.params.span,
		spans_count: item.params.spans_count,
		n1: item.params.n1,
		n2: item.params.n2,
		rise: null,
		outer_node_structure_type: item.params.outer_node_structure_type ?? form.outer_node_structure_type,
	});
	lastParams.value = currentParams();
	lastResult.value = item.result;
	clearBridgeImage();
	safetyResult.value = null;
	saveBridgeState();
	await goToTab('results');
	ElMessage.success('已应用优化方案');
	try {
		const { data } = await visualizeBridge(currentParams(), item.result);
		if (data?.image) {
			imageSrc.value = `data:image/${data.fmt || 'png'};base64,${data.image}`;
			saveBridgeImage();
			saveBridgeState();
			await goToTab('drawing');
		}
	} catch {
		ElMessage.warning('方案已应用，图纸可稍后手动生成');
	}
};

const ensureBimfaceScript = () =>
	new Promise<void>((resolve, reject) => {
		if ((window as any).BimfaceSDKLoader) return resolve();
		const existing = document.querySelector('script[data-bimface-sdk]');
		if (existing) {
			existing.addEventListener('load', () => resolve());
			existing.addEventListener('error', () => reject(new Error('Bimface SDK 加载失败')));
			return;
		}
		const script = document.createElement('script');
		script.dataset.bimfaceSdk = 'true';
		script.src = 'https://static.bimface.com/api/BimfaceSDKLoader/BimfaceSDKLoader@latest-release.js';
		script.charset = 'utf-8';
		script.onload = () => resolve();
		script.onerror = () => reject(new Error('Bimface SDK 加载失败'));
		document.head.appendChild(script);
	});

const isWebglAvailable = () => {
	try {
		const canvas = document.createElement('canvas');
		return !!(canvas.getContext('webgl') || canvas.getContext('experimental-webgl') || canvas.getContext('webgl2'));
	} catch {
		return false;
	}
};

const loadBimfaceModel = async () => {
	if (model.loaded || model.loading) return;
	model.loading = true;
	model.error = '';
	try {
		if (!isWebglAvailable()) {
			throw new Error('当前浏览器 WebGL 不可用。请使用最新版 Chrome 或 Edge，并开启浏览器硬件加速。');
		}
		const { data } = await getBimfaceViewToken();
		if (!data?.available || !data?.viewToken) {
			throw new Error('3D 模型查看器暂不可用（本地演示模式，需配置 BIMFACE 云服务密钥）');
		}
		await ensureBimfaceScript();
		await nextTick();
		const dom = bimfaceContainerRef.value;
		if (!dom) throw new Error('模型容器不存在');
		await new Promise((resolve) => requestAnimationFrame(() => requestAnimationFrame(resolve)));
		const loaderConfig = new BimfaceSDKLoaderConfig();
		loaderConfig.viewToken = data.viewToken;
		BimfaceSDKLoader.load(
			loaderConfig,
			() => {
				const webAppConfig = new Glodon.Bimface.Application.WebApplication3DConfig();
				webAppConfig.domElement = dom;
				const app = new Glodon.Bimface.Application.WebApplication3D(webAppConfig);
				app.addView(data.viewToken);
				const viewer = app.getViewer();
				viewer.addEventListener(Glodon.Bimface.Viewer.Viewer3DEvent.ViewAdded, () => {
					viewer.render();
					initLabelSystem(viewer);
				});
				model.loaded = true;
				model.loading = false;
			},
			(error: any) => {
				model.loading = false;
				model.error = `SDK 加载失败：${error || '未知错误'}`;
			}
		);
	} catch (error: any) {
		model.loading = false;
		const detail = error?.response?.data?.detail;
		if (detail === 'BIMFACE_FILE_ID is not configured' || detail?.includes('BIMFACE')) {
			model.error = '3D 模型云渲染暂未配置（需在算法服务后端配置 BIMFACE_APP_KEY、BIMFACE_APP_SECRET 与 BIMFACE_FILE_ID）';
		} else {
			model.error = detail || error?.message || '3D 模型加载失败，请检查网络或后端服务';
		}
	}
};

const retryModel = () => {
	model.loaded = false;
	model.loading = false;
	model.error = '';
	loadBimfaceModel();
};

const initLabelSystem = (viewer: any) => {
	const dom = bimfaceContainerRef.value;
	if (!dom) return;
	dom.addEventListener('mousedown', (event) => {
		const rect = dom.getBoundingClientRect();
		lastClick = { x: event.clientX - rect.left, y: event.clientY - rect.top };
	});
	dom.addEventListener('mouseup', () => {
		setTimeout(async () => {
			try {
				const selected = viewer.getSelectedComponents ? viewer.getSelectedComponents() : [];
				if (!selected?.length) return;
				const compId = String(selected[0]);
				const { data } = await getBimfaceComponentProperties(compId);
				const propObj = data.ok ? data.properties : null;
				const name = propObj?.name || `构件 #${compId}`;
				if (labelMode.value) addFloatLabel(lastClick.x, lastClick.y, name, labelCounter++);
				showInfoPanel(name, propObj);
			} catch {
				// Selection polling is best effort because BIMFace SDK versions differ.
			}
		}, 150);
	});
};

const addFloatLabel = (x: number, y: number, name: string, idx: number) => {
	const layer = labelLayerRef.value;
	const svg = labelSvgRef.value;
	const container = bimfaceContainerRef.value;
	if (!layer || !svg || !container) return;
	const color = labelColors[idx % labelColors.length];
	const width = Math.min(name.length * 14 + 24, 220);
	let left = x + 24;
	let top = y - 40;
	if (left + width > container.offsetWidth - 10) left = x - width - 24;
	if (top < 6) top = y + 14;
	const ns = 'http://www.w3.org/2000/svg';
	const line = document.createElementNS(ns, 'line');
	line.setAttribute('x1', String(x));
	line.setAttribute('y1', String(y));
	line.setAttribute('x2', String(left > x ? left : left + width));
	line.setAttribute('y2', String(top + 14));
	line.setAttribute('stroke', color);
	line.setAttribute('stroke-width', '1.5');
	line.setAttribute('stroke-dasharray', '4 2');
	line.dataset.labelId = String(idx);
	svg.appendChild(line);
	const dot = document.createElementNS(ns, 'circle');
	dot.setAttribute('cx', String(x));
	dot.setAttribute('cy', String(y));
	dot.setAttribute('r', '5');
	dot.setAttribute('fill', color);
	dot.setAttribute('stroke', 'white');
	dot.setAttribute('stroke-width', '1.5');
	dot.dataset.labelId = String(idx);
	svg.appendChild(dot);
	const tag = document.createElement('div');
	tag.dataset.labelId = String(idx);
	tag.className = 'float-label';
	tag.style.cssText = `left:${left}px;top:${top}px;width:${width}px;background:${color}`;
	tag.textContent = name;
	tag.ondblclick = () => document.querySelectorAll(`[data-label-id="${idx}"]`).forEach((el) => el.remove());
	layer.appendChild(tag);
};

const showInfoPanel = (name: string, propObj: any) => {
	propertyPanel.visible = true;
	propertyPanel.title = name;
	const rows: Array<{ key: string; value: string }> = [];
	if (propObj?.boundingBox) {
		const bb = propObj.boundingBox;
		rows.push({ key: '长度 X', value: `${Math.abs(bb.max.x - bb.min.x).toFixed(3)} m` });
		rows.push({ key: '宽度 Y', value: `${Math.abs(bb.max.y - bb.min.y).toFixed(3)} m` });
		rows.push({ key: '高度 Z', value: `${Math.abs(bb.max.z - bb.min.z).toFixed(3)} m` });
	}
	(propObj?.properties || []).forEach((group: any) => {
		(group.items || []).slice(0, 8).forEach((item: any) => rows.push({ key: item.key || item.name, value: `${item.value ?? ''}${item.unit || ''}` }));
	});
	propertyPanel.rows = rows;
};

const clearLabels = () => {
	if (labelLayerRef.value) labelLayerRef.value.innerHTML = '';
	if (labelSvgRef.value) labelSvgRef.value.innerHTML = '';
	labelCounter = 0;
	propertyPanel.visible = false;
};

const toggleLabelMode = () => {
	labelMode.value = !labelMode.value;
};

const routeTabMap: Record<string, string> = {
	chat: 'model3d',
	model3d: 'model3d',
	drawing: 'drawing',
	params: 'params',
	results: 'results',
	safety: 'safety',
	optimize: 'optimize',
};
const pathTabMap: Record<string, string> = {
	'/bridge/chat': 'model3d',
	'/bridge/model3d': 'model3d',
	'/bridge/drawing': 'drawing',
	'/bridge/params': 'params',
	'/bridge/results': 'results',
	'/bridge/safety': 'safety',
	'/bridge/optimize': 'optimize',
};
const tabPathMap = Object.fromEntries(Object.entries(pathTabMap).map(([path, tab]) => [tab, path]));

const syncTabFromRoute = () => {
	const tab = String(route.query.tab || '');
	if (routeTabMap[tab]) activeTab.value = routeTabMap[tab];
	else if (pathTabMap[route.path]) activeTab.value = pathTabMap[route.path];
};

watch(activeTab, (tab) => {
	const targetPath = tabPathMap[tab];
	if (targetPath && route.path !== targetPath) {
		const query = Object.fromEntries(Object.entries(route.query).filter(([key]) => key !== 'tab'));
		void router.replace({ path: targetPath, query });
	}
	// 仅当图纸尚未生成时才自动出图；/chat 已返回图纸或已有 imageSrc 时
	// 不再重复请求 /visualize，避免一次出图两次请求与旧图被覆盖。
	if (tab === 'drawing' && lastResult.value && !predictionIsStale.value && !imageSrc.value && !loading.visualize) {
		runVisualize();
	}
});

onMounted(() => {
	restoreBridgeState();
	syncTabFromRoute();
});
watch(() => route.fullPath, syncTabFromRoute);
watch(
	() => [
		form.total_length, form.width, form.span, form.spans_count, form.n1, form.n2, form.rise,
		form.outer_node_structure_type,
		woodType.value, serviceLifeEvidence.current_age, serviceLifeEvidence.exposure_class,
		serviceLifeEvidence.maintenance_level, serviceLifeEvidence.preservative_treatment,
	],
	saveBridgeState
);
</script>

<style scoped lang="scss">
.bridge-shell {
	height: calc(100vh - 84px);
	min-height: 720px;
	display: grid;
	grid-template-columns: minmax(0, 1fr);
	background: #f8fafc;
	color: #1e293b;
	font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, 'Helvetica Neue', Arial, 'Noto Sans', sans-serif;
	transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
}

.bridge-shell.workbench-mode {
	grid-template-columns: 380px minmax(0, 1fr);
}

/* ================= 智能对话面板 ================= */
.chat-panel {
	display: flex;
	flex-direction: column;
	background: #ffffff;
	border-right: 1px solid #e2e8f0;
	min-width: 0;
	box-shadow: 2px 0 12px rgba(15, 23, 42, 0.03);
}

.chat-header {
	height: 68px;
	padding: 0 20px;
	display: flex;
	align-items: center;
	gap: 14px;
	background: linear-gradient(135deg, #1e3a8a 0%, #2563eb 100%);
	color: white;
	box-shadow: 0 4px 16px rgba(37, 99, 235, 0.2);
}

.bridge-mark {
	width: 38px;
	height: 38px;
	display: grid;
	place-items: center;
	border-radius: 10px;
	background: rgba(255, 255, 255, 0.16);
	border: 1px solid rgba(255, 255, 255, 0.25);
	backdrop-filter: blur(8px);
	font-size: 17px;
	font-weight: 700;
	letter-spacing: 1px;
}

.chat-header h2 {
	margin: 0;
	font-size: 15px;
	font-weight: 700;
	letter-spacing: 0.3px;
}

.chat-header p {
	margin: 3px 0 0;
	font-size: 12px;
	opacity: 0.85;
	letter-spacing: 0.5px;
	font-family: 'Consolas', 'Courier New', monospace;
}

.messages {
	flex: 1;
	overflow-y: auto;
	padding: 18px 16px;
	display: flex;
	flex-direction: column;
	gap: 14px;
	background: #f8fafc;

	&::-webkit-scrollbar {
		width: 5px;
	}
	&::-webkit-scrollbar-thumb {
		background: #cbd5e1;
		border-radius: 4px;
	}
}

.msg {
	max-width: 90%;
	padding: 12px 15px;
	border-radius: 16px;
	font-size: 13.5px;
	line-height: 1.6;
	white-space: pre-wrap;
	word-break: break-word;
	transition: all 0.2s ease;
}

.msg.user {
	align-self: flex-end;
	background: linear-gradient(135deg, #2563eb 0%, #1d4ed8 100%);
	color: white;
	border-bottom-right-radius: 4px;
	box-shadow: 0 4px 14px rgba(37, 99, 235, 0.25);
}

.msg.bot {
	align-self: flex-start;
	background: #ffffff;
	color: #1e293b;
	border: 1px solid #e2e8f0;
	border-bottom-left-radius: 4px;
	box-shadow: 0 2px 8px rgba(15, 23, 42, 0.04);
}

.msg.system {
	align-self: center;
	background: #fef3c7;
	border: 1px solid #fde68a;
	color: #92400e;
	font-size: 12px;
	border-radius: 20px;
	padding: 6px 14px;
}

.msg.loading {
	color: #64748b;
	font-style: italic;
	display: flex;
	align-items: center;
	gap: 8px;
}

.chat-input-area {
	padding: 14px 16px 16px;
	border-top: 1px solid #e2e8f0;
	background: #ffffff;
}

.chat-mode-selector {
	display: flex;
	align-items: center;
	justify-content: space-between;
	gap: 10px;
	margin-bottom: 10px;

	> span {
		color: #475569;
		font-size: 13px;
		font-weight: 600;
	}

	:deep(.el-radio-group) {
		display: grid;
		grid-template-columns: repeat(2, minmax(76px, 1fr));
	}

	:deep(.el-radio-button__inner) {
		width: 100%;
		padding-inline: 12px;
	}
}

.chat-row {
	display: grid;
	grid-template-columns: 1fr 48px;
	gap: 10px;
	align-items: stretch;

	:deep(.el-textarea__inner) {
		border-radius: 8px;
		border-color: #e2e8f0;
		box-shadow: none;
		transition: all 0.2s ease;
		&:focus {
			border-color: #3b82f6;
			box-shadow: 0 0 0 3px rgba(59, 130, 246, 0.12);
		}
	}

	.el-button {
		height: auto;
		border-radius: 8px;
		background: linear-gradient(135deg, #2563eb 0%, #1d4ed8 100%);
		border: none;
		box-shadow: 0 4px 12px rgba(37, 99, 235, 0.2);
		&:hover {
			opacity: 0.95;
			transform: translateY(-1px);
		}
	}
}

/* ================= 右侧工作台主区域 ================= */
.result-panel {
	min-width: 0;
	padding: 0 16px 16px;
}

.bridge-tabs {
	height: 100%;
}

:deep(.bridge-tabs > .el-tabs__header) {
	display: none;
}

:deep(.bridge-tabs > .el-tabs__content) {
	height: 100%;
}

:deep(.el-tab-pane) {
	height: 100%;
}

.model-stage,
.drawing-stage,
.params-stage,
.results-stage,
.analysis-stage {
	position: relative;
	height: 100%;
	background: #ffffff;
	border: 1px solid #e2e8f0;
	border-radius: 12px;
	box-shadow: 0 4px 16px rgba(15, 23, 42, 0.04);
	overflow: hidden;
}

/* ================= 3D 模型舞台 ================= */
.model-stage {
	background: linear-gradient(180deg, #1e293b 0%, #0f172a 100%);
}

.bimface-container {
	position: absolute;
	inset: 0;
}

.model-placeholder,
.empty-state {
	display: flex;
	flex-direction: column;
	align-items: center;
	justify-content: center;
	gap: 14px;
	color: #64748b;
	font-size: 14px;
	background: radial-gradient(circle at center, #f8fafc 0%, #f1f5f9 100%);
	z-index: 3;
}

.model-placeholder {
	position: absolute;
	inset: 0;
	background: radial-gradient(circle at center, #1e293b 0%, #0f172a 100%);
	color: #94a3b8;

	.el-icon {
		font-size: 52px;
		color: #3b82f6;
		filter: drop-shadow(0 0 16px rgba(59, 130, 246, 0.4));
	}

	span {
		font-size: 15px;
		font-weight: 500;
		color: #cbd5e1;
	}
}

.empty-state {
	min-height: 400px;
	position: static;
	border-radius: 8px;
	border: 1px dashed #cbd5e1;
	background: #f8fafc;

	.el-icon {
		font-size: 48px;
		color: #94a3b8;
	}
}

.model-placeholder.error {
	color: #f87171;
}

.label-svg,
.label-layer {
	position: absolute;
	inset: 0;
	pointer-events: none;
	z-index: 8;
}

.label-layer {
	z-index: 9;
}

:deep(.float-label) {
	position: absolute;
	height: 30px;
	padding: 0 12px;
	border-radius: 6px;
	color: white;
	font-size: 12.5px;
	font-weight: 600;
	line-height: 30px;
	box-shadow: 0 4px 16px rgba(0, 0, 0, 0.35);
	border: 1px solid rgba(255, 255, 255, 0.4);
	backdrop-filter: blur(4px);
	pointer-events: auto;
	white-space: nowrap;
	overflow: hidden;
	text-overflow: ellipsis;
	cursor: pointer;
	transition: transform 0.15s ease;
	&:hover {
		transform: scale(1.05);
	}
}

.model-toolbar {
	position: absolute;
	top: 14px;
	right: 14px;
	z-index: 12;
	display: flex;
	gap: 8px;
	padding: 6px 10px;
	background: rgba(15, 23, 42, 0.75);
	backdrop-filter: blur(12px);
	-webkit-backdrop-filter: blur(12px);
	border: 1px solid rgba(255, 255, 255, 0.15);
	border-radius: 8px;
	box-shadow: 0 8px 24px rgba(0, 0, 0, 0.3);
}

.property-panel {
	position: absolute;
	right: 18px;
	bottom: 18px;
	z-index: 12;
	width: 330px;
	max-height: 460px;
	background: rgba(255, 255, 255, 0.96);
	backdrop-filter: blur(16px);
	-webkit-backdrop-filter: blur(16px);
	border-radius: 10px;
	box-shadow: 0 16px 36px rgba(15, 23, 42, 0.22);
	border: 1px solid #e2e8f0;
	overflow: hidden;
	animation: slideInUp 0.25s ease-out;
}

@keyframes slideInUp {
	from {
		opacity: 0;
		transform: translateY(12px);
	}
	to {
		opacity: 1;
		transform: translateY(0);
	}
}

.property-title {
	display: flex;
	align-items: center;
	justify-content: space-between;
	padding: 11px 16px;
	background: linear-gradient(135deg, #1e3a8a 0%, #2563eb 100%);
	color: white;
	font-weight: 600;
	font-size: 13.5px;
	letter-spacing: 0.3px;
}

.property-body {
	max-height: 390px;
	overflow: auto;
}

.property-row {
	display: flex;
	justify-content: space-between;
	gap: 12px;
	padding: 9px 16px;
	border-bottom: 1px solid #f1f5f9;
	font-size: 12.5px;
	transition: background 0.15s ease;

	&:hover {
		background: #f8fafc;
	}
}

.property-row span {
	color: #64748b;
}

.property-row strong {
	color: #1e293b;
}

.property-empty {
	padding: 20px;
	color: #94a3b8;
	text-align: center;
	font-size: 13px;
}

/* ================= 结构图纸舞台 ================= */
.drawing-stage,
.params-stage,
.results-stage,
.analysis-stage {
	padding: 20px;
	overflow: auto;
}

.drawing-viewer {
	height: calc(100% - 54px);
	min-height: 540px;
	overflow: auto;
	background-color: #f8fafc;
	background-image:
		radial-gradient(circle, #cbd5e1 1px, transparent 1px),
		linear-gradient(to right, #f1f5f9 1px, transparent 1px),
		linear-gradient(to bottom, #f1f5f9 1px, transparent 1px);
	background-size: 20px 20px, 100px 100px, 100px 100px;
	border: 1px solid #e2e8f0;
	border-radius: 10px;
	padding: 24px;
	box-shadow: inset 0 2px 8px rgba(15, 23, 42, 0.03);
}

.drawing-viewer img {
	display: block;
	margin: 0 auto;
	object-fit: contain;
	transform-origin: top center;
	box-shadow: 0 16px 36px rgba(15, 23, 42, 0.12);
	border: 1px solid #cbd5e1;
	border-radius: 4px;
	background: white;
	transition: transform 0.18s cubic-bezier(0.4, 0, 0.2, 1);
}

.stage-actions {
	display: flex;
	justify-content: flex-end;
	align-items: center;
	gap: 10px;
	margin-bottom: 16px;
}

/* ================= 设计参数 & 预测指标 ================= */
.params-grid {
	display: grid;
	grid-template-columns: repeat(3, minmax(180px, 1fr));
	gap: 16px 20px;
}

.params-grid span {
	margin-left: 8px;
	color: #64748b;
	font-size: 12.5px;
	font-weight: 500;
}

.metric-grid {
	display: grid;
	grid-template-columns: repeat(auto-fit, minmax(150px, 1fr));
	gap: 14px;
	margin-bottom: 18px;
}

.metric {
	border: 1px solid #e2e8f0;
	border-radius: 10px;
	padding: 16px;
	background: #f8fafc;
	transition: all 0.2s ease;

	&:hover {
		transform: translateY(-2px);
		box-shadow: 0 6px 18px rgba(15, 23, 42, 0.06);
		border-color: #cbd5e1;
	}
}

.metric span {
	display: block;
	color: #64748b;
	font-size: 12.5px;
	margin-bottom: 8px;
	font-weight: 500;
}

.metric strong {
	font-size: 24px;
	color: #2563eb;
	font-weight: 700;
	letter-spacing: -0.5px;
}

.model-trace {
	display: grid;
	grid-template-columns: 2fr 1fr 1fr;
	gap: 18px;
	padding: 14px 18px;
	margin-bottom: 18px;
	border: 1px solid #e2e8f0;
	border-radius: 8px;
	background: #f8fafc;
}

.model-trace div {
	min-width: 0;
}

.model-trace span,
.model-trace strong {
	display: block;
}

.model-trace span {
	margin-bottom: 4px;
	color: #64748b;
	font-size: 12px;
}

.model-trace strong {
	overflow-wrap: anywhere;
	color: #334155;
	font-size: 13px;
	font-weight: 600;
}

.tables {
	display: grid;
	grid-template-columns: 1fr 1fr;
	gap: 16px;
}

.tables h3 {
	margin: 0 0 12px;
	font-size: 14.5px;
	font-weight: 600;
	color: #1e293b;
}

/* ================= 安全性分析 ================= */
.wood-select {
	width: 150px;
}

.service-life-controls {
	display: grid;
	grid-template-columns: repeat(5, minmax(120px, 1fr)) auto;
	align-items: end;
	gap: 12px;
	margin-bottom: 18px;
	padding: 14px;
	background: #f8fafc;
	border: 1px solid #e2e8f0;
	border-radius: 10px;
}

.service-life-controls label {
	display: flex;
	flex-direction: column;
	gap: 6px;
	font-size: 12.5px;
	color: #475569;
	font-weight: 500;
}

.service-life-controls :deep(.el-select),
.service-life-controls :deep(.el-input-number) {
	width: 100%;
}

.service-life-controls .treatment-control {
	align-items: flex-start;
}

.safety-report {
	display: flex;
	flex-direction: column;
	gap: 16px;
}

.score-card {
	display: grid;
	grid-template-columns: repeat(4, minmax(120px, 1fr));
	gap: 16px;
	padding: 18px;
	border: 1px solid #e2e8f0;
	border-left: 5px solid;
	border-radius: 10px;
	background: #ffffff;
	box-shadow: 0 4px 16px rgba(15, 23, 42, 0.04);
}

.score-card span,
.plan-metrics span {
	display: block;
	font-size: 12.5px;
	color: #64748b;
	margin-bottom: 6px;
}

.score-card strong {
	font-size: 24px;
	font-weight: 700;
	letter-spacing: -0.5px;
}

.score-breakdown {
	display: grid;
	grid-template-columns: repeat(4, minmax(120px, 1fr));
	gap: 12px;
}

.score-breakdown div {
	padding: 14px;
	border: 1px solid #e2e8f0;
	border-radius: 10px;
	background: #f8fafc;
	transition: all 0.2s ease;
	&:hover {
		border-color: #cbd5e1;
		background: #ffffff;
	}
}

.score-breakdown span {
	display: block;
	color: #64748b;
	font-size: 12px;
	margin-bottom: 6px;
}

.score-breakdown strong {
	font-size: 19px;
	color: #2563eb;
	font-weight: 700;
}

.report-section {
	border: 1px solid #e2e8f0;
	border-radius: 10px;
	padding: 16px;
	background: #fff;
	box-shadow: 0 2px 8px rgba(15, 23, 42, 0.03);
}

.section-title {
	display: flex;
	justify-content: space-between;
	gap: 12px;
	align-items: flex-start;
	margin-bottom: 14px;
}

.section-title h3 {
	margin: 0;
	font-size: 15px;
	font-weight: 600;
	color: #1e293b;
}

.section-title span {
	max-width: 60%;
	color: #64748b;
	font-size: 12px;
	line-height: 1.6;
	text-align: right;
}

.report-note {
	margin: 12px 0 0;
	padding: 10px 14px;
	border-radius: 8px;
	background: #eff6ff;
	border: 1px solid #dbeafe;
	color: #1e40af;
	line-height: 1.6;
	font-size: 13px;
}


.structural-summary {
	display: grid;
	grid-template-columns: repeat(4, minmax(120px, 1fr));
	gap: 12px;
	margin-bottom: 12px;
}

.structural-summary div {
	padding: 12px;
	border: 1px solid #e2e8f0;
	border-radius: 10px;
	background: #f8fafc;
}

.structural-summary span {
	display: block;
	color: #64748b;
	font-size: 12px;
	margin-bottom: 6px;
}

.structural-summary strong {
	font-size: 18px;
	color: #2563eb;
	font-weight: 700;
}

.structural-summary .ok {
	border-color: #bbf7d0;
	background: #f0fdf4;
}

.structural-summary .ok strong {
	color: #16a34a;
}

.structural-summary .bad {
	border-color: #fecaca;
	background: #fef2f2;
}

.structural-summary .bad strong {
	color: #dc2626;
}

.structural-table {
	margin-top: 10px;
}

.report-grid {
	display: grid;
	grid-template-columns: minmax(0, 1fr) minmax(0, 1fr);
	gap: 16px;
}

.risk-list {
	display: flex;
	flex-direction: column;
	gap: 10px;
}

.risk-item {
	display: flex;
	justify-content: space-between;
	gap: 12px;
	padding: 12px;
	border: 1px solid #e2e8f0;
	border-radius: 8px;
	background: #f8fafc;
}

.risk-content {
	flex: 1;
	min-width: 0;
}

.risk-head {
	display: flex;
	justify-content: space-between;
	align-items: center;
	gap: 10px;
	margin-bottom: 6px;
}

.risk-item strong {
	display: block;
	margin-bottom: 4px;
	font-size: 13.5px;
	color: #1e293b;
}

.risk-item ul {
	margin: 4px 0 8px;
	padding-left: 18px;
	color: #64748b;
	line-height: 1.6;
	font-size: 12.5px;
}

.risk-item span {
	color: #64748b;
	line-height: 1.6;
	font-size: 12.5px;
}

.life-summary {
	display: grid;
	grid-template-columns: repeat(3, 1fr);
	gap: 10px;
	margin-bottom: 14px;
	color: #475569;
}

.life-summary div {
	padding: 10px 12px;
	border-radius: 8px;
	background: #f8fafc;
	border: 1px solid #e2e8f0;
	font-size: 12.5px;
}

.confidence-note {
	margin-bottom: 14px;
	color: #64748b;
	font-size: 13px;
	line-height: 1.6;
}

/* ================= Pareto 方案优化 ================= */
.optimize-form {
	display: flex;
	justify-content: space-between;
	align-items: flex-end;
	gap: 16px;
	padding: 16px;
	margin-bottom: 16px;
	border: 1px solid #e2e8f0;
	border-radius: 10px;
	background: #ffffff;
	box-shadow: 0 2px 8px rgba(15, 23, 42, 0.03);
}

.optimize-form .el-form {
	flex: 1;
}

.optimize-inputs {
	display: grid;
	grid-template-columns: repeat(4, minmax(150px, 1fr));
	gap: 14px;
}

.optimize-inputs span {
	margin-left: 8px;
	color: #64748b;
	font-size: 12px;
}

.optimize-summary {
	display: flex;
	justify-content: space-between;
	gap: 12px;
	margin-bottom: 14px;
	color: #64748b;
	font-size: 13px;
	font-weight: 500;
}

.original-plan-panel {
	margin-bottom: 16px;
	padding: 16px 20px;
	border: 1px solid #bfdbfe;
	border-left: 5px solid #2563eb;
	border-radius: 10px;
	background: linear-gradient(135deg, #f8fafc 0%, #eff6ff 100%);
	box-shadow: 0 4px 16px rgba(37, 99, 235, 0.06);
}

.original-title {
	display: flex;
	justify-content: space-between;
	align-items: flex-start;
	gap: 12px;
	margin-bottom: 14px;
}

.original-title h3 {
	margin: 0 0 4px;
	font-size: 16px;
	font-weight: 700;
	color: #1e3a8a;
}

.original-title span,
.original-grid span {
	color: #64748b;
	font-size: 12px;
}

.original-title strong {
	color: #2563eb;
	font-size: 19px;
	font-weight: 700;
	white-space: nowrap;
}

.original-grid {
	display: grid;
	grid-template-columns: repeat(auto-fit, minmax(120px, 1fr));
	gap: 10px;
}

.original-grid div {
	padding: 10px 12px;
	border-radius: 8px;
	background: #ffffff;
	border: 1px solid #e2e8f0;
	min-width: 0;
	box-shadow: 0 1px 4px rgba(15, 23, 42, 0.02);
}

.original-grid strong {
	display: block;
	margin-top: 5px;
	color: #1e293b;
	font-size: 15px;
	font-weight: 700;
	white-space: nowrap;
}

.optimize-grid {
	display: grid;
	grid-template-columns: repeat(3, minmax(260px, 1fr));
	gap: 16px;
}

.plan-card {
	border: 1px solid #e2e8f0;
	border-radius: 12px;
	padding: 16px;
	background: #ffffff;
	display: flex;
	flex-direction: column;
	gap: 14px;
	box-shadow: 0 4px 14px rgba(15, 23, 42, 0.04);
	transition: all 0.22s cubic-bezier(0.4, 0, 0.2, 1);

	&:hover {
		transform: translateY(-3px);
		box-shadow: 0 12px 28px rgba(15, 23, 42, 0.09);
		border-color: #93c5fd;
	}
}

.plan-head {
	display: flex;
	justify-content: space-between;
	align-items: flex-start;
	gap: 10px;
	padding-left: 10px;
	border-left: 5px solid #2563eb;
}

.plan-head h3 {
	margin: 0 0 4px;
	font-size: 15.5px;
	font-weight: 700;
	color: #1e293b;
}

.plan-head span {
	color: #64748b;
	font-size: 12.5px;
}

.delta-pill {
	padding: 4px 10px;
	border-radius: 999px;
	font-size: 12px;
	font-weight: 700;
	white-space: nowrap;
}

.plan-metrics {
	display: grid;
	grid-template-columns: repeat(4, 1fr);
	gap: 8px;
}

.plan-metrics div {
	padding: 10px 8px;
	border-radius: 8px;
	background: #f8fafc;
	border: 1px solid #f1f5f9;
	text-align: center;
}

.plan-metrics strong {
	display: block;
	font-size: 17px;
	color: #2563eb;
	font-weight: 700;
}

.plan-metrics em,
.plan-detail em {
	display: inline-block;
	margin-top: 4px;
	font-style: normal;
	font-size: 11.5px;
	font-weight: 700;
	padding: 2px 6px;
	border-radius: 4px;
}

.up {
	color: #047857 !important;
	background: #ecfdf5 !important;
	border: 1px solid #a7f3d0;
}

.down {
	color: #be123c !important;
	background: #fff1f2 !important;
	border: 1px solid #fecdd3;
}

.same {
	color: #64748b !important;
	background: #f1f5f9 !important;
	border: 1px solid #e2e8f0;
}

.plan-detail {
	display: grid;
	grid-template-columns: repeat(2, minmax(0, 1fr));
	gap: 8px;
	font-size: 13px;
}

.plan-detail div {
	padding: 10px 12px;
	border-radius: 8px;
	background: #f8fafc;
	border: 1px solid #e2e8f0;
	min-width: 0;
}

.plan-detail span {
	display: block;
	margin-bottom: 5px;
	color: #64748b;
	font-size: 12px;
}

.plan-detail strong {
	display: block;
	color: #1e293b;
	font-weight: 600;
	word-break: break-word;
}

/* ================= 响应式适配 ================= */
@media (max-width: 1100px) {
	.bridge-shell,
	.bridge-shell.workbench-mode {
		grid-template-columns: 1fr;
		height: auto;
	}
	.chat-panel {
		min-height: 520px;
	}
	.result-panel {
		min-height: 720px;
	}
	.model-trace {
		grid-template-columns: 1fr;
		gap: 10px;
	}
}

@media (max-width: 1280px) {
	.optimize-grid,
	.original-grid,
	.report-grid {
		grid-template-columns: 1fr;
	}

	.optimize-form {
		align-items: stretch;
		flex-direction: column;
	}

	.optimize-inputs,
	.score-breakdown {
		grid-template-columns: repeat(2, minmax(0, 1fr));
	}
}
</style>
