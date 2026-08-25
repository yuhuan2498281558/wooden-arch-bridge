import axios from 'axios';

const bridgeApi = axios.create({
	baseURL: import.meta.env.VITE_BRIDGE_API_URL || 'http://127.0.0.1:8001',
	timeout: 60000,
	headers: {
		'Content-Type': 'application/json',
	},
});

export type OuterNodeStructureType = 'front_half' | 'back_half';

export interface BridgeParams {
	total_length: number;
	width: number;
	span: number;
	spans_count: number;
	n1: number;
	n2: number;
	rise?: number | null;
	outer_node_structure_type?: OuterNodeStructureType | null;
}

export interface ServiceLifeEvidence {
	current_age: number;
	exposure_class: 'protected' | 'sheltered' | 'exposed';
	maintenance_level: 'good' | 'regular' | 'poor';
	preservative_treatment: boolean;
}

export function predictBridge(params: BridgeParams) {
	return bridgeApi.post('/predict', {
		length: params.total_length,
		width: params.width,
		span: params.span,
		spans_count: params.spans_count,
		n1: params.n1,
		n2: params.n2,
		rise: params.rise || null,
		outer_node_structure_type: params.outer_node_structure_type || null,
	});
}

export function chatBridge(data: { message: string; history: any[]; outer_node_structure_type?: OuterNodeStructureType }) {
	return bridgeApi.post('/chat', data);
}

export function visualizeBridge(params: BridgeParams, result: any) {
	return bridgeApi.post('/visualize', {
		params,
		result,
		fmt: 'png',
	});
}

export function getBimfaceViewToken() {
	return bridgeApi.get('/bimface/view-token');
}

export function getBimfaceComponentProperties(componentId: string) {
	return bridgeApi.get(`/bimface/component/${componentId}/properties`);
}

export function analyzeBridge(params: BridgeParams, result: any, woodType: string, serviceLifeEvidence: ServiceLifeEvidence) {
	return bridgeApi.post('/analyze', {
		params,
		result,
		wood_type: woodType,
		service_life_evidence: serviceLifeEvidence,
	});
}

export function optimizeBridge(
	params: Pick<BridgeParams, 'span' | 'width' | 'spans_count' | 'total_length' | 'n1' | 'n2' | 'rise' | 'outer_node_structure_type'>
) {
	return bridgeApi.post('/optimize', params);
}

export function exportBridgeExcel(params: BridgeParams, result: any) {
	return bridgeApi.post(
		'/export/excel',
		{ params, result },
		{ responseType: 'blob' }
	);
}
