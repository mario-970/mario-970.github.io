---
layout: page
title: 算子日报
permalink: /operator-daily/
---

# 算子日报

每日汇总 CUDA 开源仓 release note、芯片动态与 arXiv 算子内核相关论文。

## 历史日报

{% for post in site.tags['算子日报'] %}
- [{{ post.title }}]({{ post.url }}) — {{ post.date | date: "%Y-%m-%d" }}
{% endfor %}
