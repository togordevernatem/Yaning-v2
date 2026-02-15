#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
YY.py - 生成 SE-GC-TPP 完整流程图 (修复所有语法错误)
"""
import graphviz


def generate_se_gc_tpp_graph():
    # 初始化 Graphviz Digraph，禁用HTML标签解析，避免语法错误
    dot = graphviz.Digraph(
        'SE-GC-TPP_Flowchart',
        comment='SE-GC-TPP Pipeline',
        format='png',
        graph_attr={
            'rankdir': 'LR',
            'fontname': 'Arial',
            'fontsize': '10',
            'bgcolor': 'white',
            'splines': 'ortho'  # 正交边，布局更清晰
        },
        node_attr={
            'fontname': 'Arial',
            'fontsize': '9',
            'shape': 'box'  # 默认节点形状
        },
        edge_attr={
            'fontname': 'Arial',
            'fontsize': '8',
            'arrowsize': '0.8'
        }
    )

    # -----------------------------
    # 样式定义 (无重复参数，兼容Graphviz)
    # -----------------------------
    # 定义样式字典，避免重复赋值
    styles = {
        'infra': {'style': 'filled', 'fillcolor': '#eceff1', 'color': '#455a64', 'penwidth': '2'},
        'phase1': {'style': 'filled', 'fillcolor': '#fff3e0', 'color': '#ff9800', 'penwidth': '2'},
        'llm': {'style': 'filled,dashed', 'fillcolor': '#ffebee', 'color': '#d32f2f', 'penwidth': '3'},
        'profile': {'style': 'filled', 'fillcolor': '#fff9c4', 'color': '#fbc02d', 'penwidth': '3'},
        'model': {'style': 'filled', 'fillcolor': '#e8eaf6', 'color': '#3f51b5', 'penwidth': '2'},
        'loss': {'style': 'filled', 'fillcolor': '#fbe9e7', 'color': '#ff5722', 'penwidth': '2'},
        'openenc': {'style': 'filled,dashed', 'fillcolor': '#f3e5f5', 'color': '#9c27b0', 'penwidth': '2'}
    }

    # -----------------------------
    # 区域 0: 数据基础设施
    # -----------------------------
    with dot.subgraph(name='cluster_Infra') as infra:
        infra.attr(label='Data Infrastructure (2005-2015)', rank='same')
        infra.node('RawDS', 'ICEWS Dataset', shape='ellipse', **styles['infra'])
        infra.node('Split', 'Strict Split\n(No Leakage)', shape='diamond', **styles['infra'])
        infra.node('TrainH', 'Train History\nH_tr', shape='ellipse', **styles['infra'])
        infra.edge('RawDS', 'Split')
        infra.edge('Split', 'TrainH', label='70% Train')

    # -----------------------------
    # 区域 1: 离线画像构建 (Phase 1: DIMVP++)
    # -----------------------------
    with dot.subgraph(name='cluster_Phase1') as phase1:
        phase1.attr(label='Phase 1: Offline DIMVP++ Construction (Train-Only)', rank='same')

        # LLM Agent (无HTML标签，纯文本换行)
        phase1.node('LLM', 'LLM Agent (Frozen)\nRole: Prior & Compression\nTemp=0, Cached, No Ext Facts',
                    shape='parallelogram', **styles['llm'])

        # View A: Anchor
        with phase1.subgraph(name='cluster_ViewA') as viewA:
            viewA.attr(label='View A: Anchor')
            viewA.node('Counts', 'Raw Type Counts\ncnt(c)', **styles['phase1'])
            viewA.node('Dirichlet', 'Dirichlet Smooth\nα > 0', **styles['phase1'])
            viewA.edge('Counts', 'Dirichlet')

        # View B: Momentum
        with phase1.subgraph(name='cluster_ViewB') as viewB:
            viewB.attr(label='View B: Momentum')
            viewB.node('PriorLambda', 'Prior λ_hat', **styles['phase1'])
            viewB.node('InitTheta', 'Init θ & Reg Target', **styles['phase1'])
            viewB.node('LearnDecay', 'Learnable Decay λ\nsoftplus(θ)', **styles['phase1'])
            viewB.node('WeightedCnt', 'Weighted Count\nexp(-λΔt)', **styles['phase1'])
            viewB.node('PostSmooth', 'Post-Smoothing\nwith View A', **styles['phase1'])
            viewB.edge('PriorLambda', 'InitTheta')
            viewB.edge('InitTheta', 'LearnDecay', style='dashed')
            viewB.edge('LearnDecay', 'WeightedCnt')
            viewB.edge('WeightedCnt', 'PostSmooth')

        # View C: Semantic
        with phase1.subgraph(name='cluster_ViewC') as viewC:
            viewC.attr(label='View C: Semantic')
            viewC.node('Codebook', 'Codebook Defs\ndef(c)', **styles['phase1'])
            viewC.node('Tmpl', 'Train Templates\nT_u_tr', **styles['phase1'])
            viewC.node('Summary', 'Controlled Summary\nTrain-only facts', **styles['phase1'])
            viewC.node('OpenEnc1', 'Open Encoder φ\n(SBERT/MiniLM)', shape='parallelogram', **styles['openenc'])
            viewC.node('OpenEnc2', 'Open Encoder ψ\n(SBERT/MiniLM)', shape='parallelogram', **styles['openenc'])
            viewC.node('MixCode', 'Codebook Mix\n(Weighted by A)', **styles['phase1'])
            viewC.node('ConcatC', 'Fuse\nConcat & Project', **styles['phase1'])

            viewC.edge('Codebook', 'OpenEnc1')
            viewC.edge('Tmpl', 'Summary')
            viewC.edge('Summary', 'OpenEnc2')
            viewC.edge('OpenEnc1', 'MixCode')
            viewC.edge('MixCode', 'ConcatC')
            viewC.edge('OpenEnc2', 'ConcatC')

        # Phase 1 内部连线
        phase1.edge('TrainH', 'Counts')
        phase1.edge('TrainH', 'WeightedCnt')
        phase1.edge('TrainH', 'Tmpl')
        phase1.edge('Dirichlet', 'PostSmooth', style='dashed')
        phase1.edge('Dirichlet', 'MixCode', style='dashed')
        phase1.edge('LLM', 'PriorLambda', style='dashed', label='Define λ')
        phase1.edge('LLM', 'Summary', style='dashed', label='Summarize')

    # -----------------------------
    # 桥梁: 静态画像矩阵 P
    # -----------------------------
    dot.node('ProfileP', 'Static Profile Matrix P\nConcat(A, B, C)\nDim: N x D_p',
             shape='parallelogram', **styles['profile'])
    dot.edge('Dirichlet', 'ProfileP')
    dot.edge('PostSmooth', 'ProfileP')
    dot.edge('ConcatC', 'ProfileP')

    # -----------------------------
    # 区域 2: 在线建模 (Phase 2: SE-GC-TPP)
    # -----------------------------
    with dot.subgraph(name='cluster_Phase2') as phase2:
        phase2.attr(label='Phase 2: Online SE-GC-TPP Modeling (Dynamic)', rank='same')
        phase2.node('Snapshot', 'Graph Snapshot\nG(t)', shape='ellipse', **styles['model'])
        phase2.node('TimeDelta', 'Time Δt', shape='ellipse', **styles['model'])
        phase2.node('Inject', 'Feature Injection\nConcat [ X_t || P_u ]', shape='diamond', **styles['model'])
        phase2.node('GConvGRU', 'GConvGRU\nDynamic Encoder', **styles['model'])
        phase2.node('TimeEnc', 'Time Encoder', **styles['model'])
        phase2.node('Fusion', 'Fusion MLP', **styles['model'])
        phase2.node('Head', 'TPP Head\nLogNormal(μ, σ)', shape='ellipse', **styles['model'])

        phase2.edge('Snapshot', 'Inject')
        phase2.edge('Inject', 'GConvGRU')
        phase2.edge('GConvGRU', 'Fusion')
        phase2.edge('TimeDelta', 'TimeEnc')
        phase2.edge('TimeEnc', 'Fusion')
        phase2.edge('Fusion', 'Head')

    # 连接 P 到 Phase 2
    dot.edge('ProfileP', 'Inject', label='Static Lookup', style='bold')

    # -----------------------------
    # 损失函数 (Loss)
    # -----------------------------
    dot.node('NLL', 'NLL Loss\nMain Task', **styles['loss'])
    dot.node('RegLoss', 'Reg Loss\nγ(λ - λ_hat)²', **styles['loss'])
    dot.node('TotalLoss', 'Total Loss', shape='ellipse', **styles['loss'])

    dot.edge('Head', 'NLL')
    dot.edge('InitTheta', 'RegLoss', style='dashed', label='Regularization')
    dot.edge('NLL', 'TotalLoss')
    dot.edge('RegLoss', 'TotalLoss')

    # -----------------------------
    # 保存并生成图片
    # -----------------------------
    output_path = 'se_gc_tpp_flowchart'
    try:
        # 保存到当前目录，不自动打开（避免系统环境问题）
        dot.render(output_path, directory='./', view=False)
        print(f"✅ 流程图生成成功！文件路径: {output_path}.png")
    except Exception as e:
        print(f"❌ 生成失败: {e}")


if __name__ == '__main__':
    generate_se_gc_tpp_graph()
