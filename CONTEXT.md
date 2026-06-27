# Research Gap Detection Agent

Contexto de dominio do agente que identifica lacunas de pesquisa a partir de artigos cientificos. Este documento fixa a linguagem usada para separar lacunas reais, candidatas e evidencias no corpus analisado.

## Language

**Lacuna**:
Pergunta de pesquisa ainda nao explorada.
_Avoid_: gap sem qualificacao, oportunidade generica, tema interessante

**Lacuna candidata atual**:
Postulante de lacuna que parece ainda nao explorada no corpus disponivel ate a data de corte analisada.
_Avoid_: lacuna comprovada, descoberta definitiva

**Data de corte**:
Data do artigo mais recente entre os artigos ranqueados usados como corpus da analise.
_Avoid_: data de execucao, hoje absoluto

**Evidencia textual de lacuna candidata**:
Sinal extraido dos artigos que sustenta uma **Lacuna candidata atual** sem usar a analise do grafo.
_Avoid_: intuicao da LLM, evidencia do grafo

**Dados estruturados do artigo**:
Conjunto de respostas extraidas e metadados de um artigo usados como entrada para identificar **Lacunas candidatas atuais**.
_Avoid_: artigo inteiro, texto bruto sem estrutura, apenas metadados

**Insight estruturado de artigo**:
Unidade de entrada da etapa textual que combina metadados e extracoes de um unico artigo.
_Avoid_: listas separadas de metadados e extracoes

**Forca de evidencia textual**:
Pontuacao que expressa quao bem uma **Lacuna candidata atual** e sustentada pelas evidencias textuais do corpus observado.
_Avoid_: probabilidade de verdade, certeza cientifica, metrica externa de avaliacao do agente

**Contraevidencia textual**:
Sinal no corpus observado que enfraquece ou limita uma **Lacuna candidata atual**.
_Avoid_: erro do agente, descarte automatico

**Lacuna candidata final**:
Postulante de lacuna apresentado na resposta final apos combinacao entre evidencias textuais e sinais do grafo.
_Avoid_: secao isolada de grafo, secao isolada de texto

**Origem da evidencia final**:
Classificacao que indica se uma **Lacuna candidata final** veio de evidencia textual, do grafo, ou de ambos.
_Avoid_: origem implicita, mistura opaca de sinais, graph_only

**Nicho do grafo**:
Combinacao de conceitos, subareas ou topicos relevantes no corpus que aparecem pouco conectados ou nao explorados em conjunto no grafo.
_Avoid_: lacuna textual, cluster generico, tema amplo

## Relationships

- Uma **Lacuna candidata atual** pode representar uma **Lacuna**, mas ainda precisa ser apresentada como hipotese.
- Uma **Lacuna candidata atual** depende do corpus disponivel e da data de corte da analise.
- A **Data de corte** define ate quando uma **Lacuna candidata atual** parece nao explorada no corpus observado.
- A LLM chamada pelo `gap_identifier` calcula a **Data de corte** como a maior `published_date` entre os **Insights estruturados de artigo** recebidos.
- Quando houver **Insights estruturados de artigo**, `cutoff_date` deve ser nao nulo; o no rejeita uma resposta estruturada que omita essa data.
- Quando nenhum **Insight estruturado de artigo** for recebido, o `gap_identifier` deve retornar resultado vazio, `cutoff_date` nulo e um aviso estruturado; ausencia de entrada nao e evidencia de ausencia de lacunas.
- A etapa de extracao por artigo deve produzir um **Insight estruturado de artigo** com `paper_id`, `title`, `published_date`, `questions_answered`, `methodologies`, `not_addressed` e `stated_limitations`.
- O `paper_extractor` preserva os documentos convertidos com `full_text` em `extracted_documents` e armazena separadamente os contratos `ExtractedInsights` em `extracted`.
- O contrato minimo atual de `extracted` inicializa as quatro listas analiticas vazias e nao interpreta semanticamente o `full_text`; essa extracao semantica futura permanece fora do PRD atual.
- As listas de um **Insight estruturado de artigo** podem estar vazias; isso indica ausencia de evidencia extraida para o campo, nao um insight invalido.
- O `gap_identifier` deve receber **Insights estruturados de artigo** e nao deve ser responsavel por interpretar o `full_text` dos artigos.
- `title` e `published_date` devem fazer parte diretamente do **Insight estruturado de artigo**, evitando uma consulta paralela aos artigos por `paper_id`.
- Na versao inicial, o `gap_identifier` deve enviar o conjunto completo de **Insights estruturados de artigo** em uma unica requisicao para analise comparativa pela LLM.
- A skill do `gap_identifier` define as instrucoes da analise; o no `gap_identifier` prepara o prompt, chama a LLM e valida sua resposta estruturada.
- O no `gap_identifier` deve preservar os **Insights estruturados de artigo** originais diretamente no estado e armazenar separadamente as **Lacunas candidatas atuais**, permitindo auditoria sem pedir que a LLM repita ou reescreva a entrada.
- A saida estruturada do `gap_identifier` deve ser um `GapIdentificationResult` com `cutoff_date`, `warnings` e `gaps`.
- Cada aviso do `GapIdentificationResult` deve ser um `GapWarning` com `code` estavel e `message` legivel; os codigos iniciais sao `no_extracted_insights`, `invalid_counter_evidence_reference`, `invalid_evidence_reference` e `missing_evidence`.
- O schema tecnico das lacunas continuara se chamando `IdentifiedGap`, embora cada instancia represente semanticamente uma **Lacuna candidata atual**, nao uma lacuna definitiva.
- O `FinalReport` tambem deve manter `IdentifiedGap` como schema de suas lacunas; nao deve ser criado um segundo modelo `FinalCandidateGap` apenas para a etapa de agregacao.
- `IdentifiedGap` deve conter `research_question`, `description`, `evidence_strength`, `evidence`, `rationale` e `counter_evidence`, alem dos campos opcionais de fusao `origin`, `matched_graph_hypothesis` e `graph_refinement`.
- Cada `GapEvidence.evidence_type` deve aceitar somente `stated_limitations`, `recurring_not_addressed` e `contrast`; `IdentifiedGap` nao deve duplicar esses valores em um campo `evidence_types`.
- Os artigos de suporte de uma `IdentifiedGap` devem ser derivados de `evidence[].paper_id`; nao deve existir um campo duplicado `supporting_paper_ids`.
- Cada **Contraevidencia textual** deve conter `paper_id` e `description`, permitindo rastrear qual artigo contradiz ou responde parcialmente a candidata.
- `CounterEvidence.description` deve ser uma parafrase fiel dos **Insights estruturados de artigo**, sem citacao inventada nem informacao adicional.
- Uma contraevidencia com `paper_id` inexistente deve ser removida individualmente e gerar `invalid_counter_evidence_reference`, sem descartar automaticamente a `IdentifiedGap`.
- `evidence` deve ser uma lista de `GapEvidence`, cada item com `paper_id`, `evidence_type` e `description`, em vez de um texto unico.
- `GapEvidence.description` deve ser uma parafrase fiel da evidencia presente nos **Insights estruturados de artigo**, sem citacao inventada nem informacao adicional.
- Toda `IdentifiedGap` deve conter pelo menos um `GapEvidence`; evidencia com artigo inexistente ou lista vazia invalida somente a `IdentifiedGap` afetada e gera um aviso estruturado.
- O calculo conceitual de `evidence_strength` usa a escala de 0 a 100, mas a saida qualificada aceita somente inteiros de 70 a 100; a **Data de corte** pertence ao `GapIdentificationResult` e nao deve ser repetida em cada `IdentifiedGap`.
- O estado deve armazenar essa saida em `gap_identification`, substituindo a lista isolada `content_gaps`; o `aggregator` consome o objeto completo.
- Uma **Lacuna candidata atual** deve ser apresentada de forma auditavel, incluindo a pergunta de pesquisa postulada, sua **Forca de evidencia textual**, os tipos de evidencia, a **Data de corte**, a justificativa e eventuais **Contraevidencias textuais**.
- Uma **Evidencia textual de lacuna candidata** pode vir de limitacoes declaradas por autores, perguntas ou escopos marcados como nao abordados por mais de um artigo, ou contraste entre perguntas respondidas e uma pergunta vizinha ainda nao abordada.
- Evidencia do grafo nao sustenta sozinha a etapa textual de identificacao; ela entra apenas na agregacao final.
- A etapa textual deve retornar todas as **Lacunas candidatas atuais** cuja **Forca de evidencia textual** ultrapasse o limiar definido, em vez de retornar uma quantidade fixa de candidatas.
- A **Forca de evidencia textual** combina recorrencia entre artigos independentes, qualidade do sinal textual e coerencia tematica com as perguntas respondidas e os escopos nao abordados.
- A **Forca de evidencia textual** nao inclui evidencia do grafo nem impacto esperado da candidata.
- A **Forca de evidencia textual** deve ser calculada pela LLM a partir dos criterios definidos na skill; o codigo valida o contrato da resposta, mas nao recalcula o score.
- Uma futura divisao da analise em varias chamadas de LLM depende de problemas observados de contexto ou qualidade e nao faz parte da versao inicial.
- O limiar inicial de **Forca de evidencia textual** para uma candidata sair da etapa textual e 70/100, podendo ser ajustado depois por avaliacao empirica.
- **Contraevidencia textual** reduz a **Forca de evidencia textual** e deve aparecer na candidata, mas so elimina uma **Lacuna candidata atual** quando o corpus observado ja responde claramente a pergunta postulada.
- A agregacao final deve produzir uma lista unica de **Lacunas candidatas finais**, em vez de separar a resposta em uma lista textual e outra lista do grafo.
- Cada **Lacuna candidata final** deve declarar sua **Origem da evidencia final** como textual_only ou textual_and_graph; o campo e obrigatorio em toda lacuna final.
- Quando houver fusao, `matched_graph_hypothesis` preserva o dict inteiro da hipotese ranqueada usada e `graph_refinement` explica como ela refinou a pergunta textual. Em `textual_only`, ambos permanecem nulos.
- A lista final deve priorizar **Lacunas candidatas finais** com origem textual_and_graph, depois textual_only; dentro de cada grupo, deve usar a forca do sinal disponivel.
- A funcao central da agregacao final e fundir nichos achados pelo grafo com possiveis lacunas identificadas nos artigos, consolidando uma **Lacuna candidata final** quando os dois sinais apontam para uma direcao de pesquisa compativel.
- A agregacao final deve ser descrita principalmente como uma etapa de fusao de evidencias; sua funcao de governanca surge das restricoes impostas a essa fusao, nao de uma revisao critica independente.
- Um **Nicho do grafo** representa areas que parecem relevantes mas pouco exploradas em conjunto, servindo como contexto de fusao para possiveis lacunas identificadas nos artigos.
- Quando uma **Lacuna candidata atual** e fundida com um **Nicho do grafo**, o resultado deve refinar a pergunta de pesquisa final em vez de apenas somar evidencias ou aumentar prioridade.
- Um **Nicho do grafo** nao pode gerar sozinho uma **Lacuna candidata final**; ele so pode refinar ou fortalecer uma candidata sustentada por evidencia textual.
- Uma **Lacuna candidata atual** que passou pelo limiar textual deve entrar na resposta final mesmo sem **Nicho do grafo** correspondente, como origem textual_only.
- A agregacao final so pode descartar uma **Lacuna candidata atual** que passou pelo limiar textual quando ela for duplicata semantica de outra candidata mais forte ou quando houver contraevidencia clara de que a pergunta ja foi respondida.
- Na versao inicial, cada hipotese ranqueada produzida pelo grafo representa um **Nicho do grafo** para fins de agregacao e fornece diretamente seus conceitos, relacoes ausentes, sinais topologicos e pontuacoes.
- A LLM do `aggregator` recebe somente as hipoteses ranqueadas do grafo; o resumo geral e as demais metricas agregadas do grafo nao participam da fusao inicial.
- O match entre uma **Lacuna candidata atual** e um **Nicho do grafo** deve exigir sobreposicao ou equivalencia de conceitos centrais, compatibilidade com a relacao pouco explorada no grafo e uma justificativa de como o nicho refina a pergunta.
- Um **Nicho do grafo** apenas genericamente relacionado ao tema nao basta para fusao; nesses casos, a candidata deve permanecer textual_only.
- O relatorio final deve explicitar que as lacunas apresentadas sao candidatas: hipoteses de perguntas ainda nao exploradas no corpus analisado ate a **Data de corte**.
- Cada **Lacuna candidata final** apresentada no relatorio deve incluir pergunta de pesquisa, origem da evidencia final, forca de evidencia textual, **Data de corte**, evidencias principais com artigos, nicho do grafo quando houver fusao, contraevidencias ou ressalvas, e justificativa curta.

## Example dialogue

> **Dev:** "Podemos dizer que o agente encontrou uma **Lacuna**?"
> **Domain expert:** "Na resposta final, diga que encontrou uma **Lacuna candidata atual**; ela so vira **Lacuna** se houver confirmacao mais forte de que a pergunta realmente nao foi explorada."

## Flagged ambiguities

- "lacuna" estava sendo usado tanto para pergunta realmente nao explorada quanto para hipotese gerada pelo agente; resolvido: **Lacuna** e a pergunta nao explorada, enquanto **Lacuna candidata atual** e o postulante produzido pelo sistema.
- "atual" poderia significar a data de execucao do agente ou a data do corpus; resolvido: em **Lacuna candidata atual**, atual significa ate a **Data de corte** do corpus observado.
- "evidencia de lacuna" poderia misturar sinais textuais e sinais do grafo; resolvido: a etapa textual usa apenas **Evidencia textual de lacuna candidata**, e o grafo e combinado depois.
- "limite de lacunas" poderia significar top N fixo; resolvido: o limite principal e um limiar minimo de **Forca de evidencia textual**.
- "entrada do grafo no aggregator" poderia significar todo o resultado do grafo ou nichos previamente normalizados; resolvido: na versao inicial, significa somente as hipoteses ranqueadas produzidas pelo grafo.
