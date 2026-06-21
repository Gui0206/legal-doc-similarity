"""Gerador de corpus jurídico sintético com pares rotulados.

Para cada "caso" criamos um documento-base e variantes na MESMA caso_id
(o enunciado diz: todo documento pertence a exatamente um caso, e comparamos
dentro do caso). As variantes cobrem os cenários que o classificador precisa
distinguir:

  copia            -> cópia exata (mesmo texto)
  copia (ocr)      -> cópia com ruído de OCR / espaçamento
  versao           -> mesma tese, mas nomes/valores/datas/parágrafos trocados
  diferente (area) -> MESMO boilerplate + MESMO vocabulário jurídico, conteúdo
                      distinto  ==> a "armadilha do boilerplate"
  diferente (outro)-> assunto totalmente diferente

Tudo determinístico (seed fixa) para ser reproduzível pelo avaliador.
"""
from __future__ import annotations

import random
from dataclasses import dataclass

# ---- Boilerplate jurídico compartilhado por (quase) todos os documentos ----
# Propositalmente extenso: peças jurídicas reais são dominadas por linguagem
# padrão. Esse domínio faz a similaridade de vocabulário (e de embeddings)
# SATURAR -> e e por isso que o eixo semantico sozinho nao distingue um
# documento diferente-mas-da-mesma-area de uma versao.
PETICAO_HEADER = (
    "EXCELENTISSIMO SENHOR DOUTOR JUIZ DE DIREITO DA VARA CIVEL DA COMARCA. "
    "Por seu advogado que esta subscreve, vem respeitosamente a presenca de "
    "Vossa Excelencia, com fundamento no artigo 319 do Codigo de Processo Civil, "
    "propor a presente acao, pelos fatos e fundamentos juridicos a seguir expostos. "
    "Preliminarmente, requer a concessao dos beneficios da justica gratuita, nos "
    "termos da lei, por nao possuir a parte autora condicoes de arcar com as custas "
    "processuais sem prejuizo do proprio sustento e de sua familia. Requer ainda a "
    "inversao do onus da prova, ante a hipossuficiencia tecnica e probatoria da parte."
)
PETICAO_DIREITO = (
    "DO DIREITO. O direito do autor encontra amparo na legislacao vigente e na "
    "jurisprudencia consolidada dos tribunais superiores. A responsabilidade civil "
    "exige a presenca de conduta, dano e nexo de causalidade, todos demonstrados nos autos. "
    "Cumpre destacar que a relacao juridica em exame submete-se ao regime protetivo "
    "aplicavel, impondo-se a reparacao integral dos danos suportados pela parte autora. "
    "A jurisprudencia e pacifica no sentido de reconhecer o dever de indenizar sempre "
    "que presentes os pressupostos legais, conforme reiterados precedentes dos tribunais."
)
PETICAO_PEDIDO = (
    "DOS PEDIDOS. Diante do exposto, requer a Vossa Excelencia se digne a julgar "
    "totalmente procedente a presente demanda, condenando a parte requerida ao "
    "pagamento das verbas pleiteadas, acrescidas de juros e correcao monetaria, "
    "bem como das custas processuais e honorarios advocaticios. Requer a citacao da "
    "parte requerida para, querendo, apresentar contestacao no prazo legal, sob pena "
    "de revelia e confissao quanto a materia de fato. Protesta provar o alegado por "
    "todos os meios de prova em direito admitidos, especialmente prova documental, "
    "testemunhal e pericial. Da-se a causa o valor indicado. Termos em que pede deferimento."
)
SENTENCA_HEADER = (
    "Vistos e examinados os presentes autos. RELATORIO. Trata-se de acao submetida "
    "a este juizo, devidamente distribuida e processada na forma da lei, "
    "tendo sido oportunizado o contraditorio e a ampla defesa as partes."
)
SENTENCA_DISPOSITIVO = (
    "DISPOSITIVO. Ante o exposto, com fulcro no artigo 487, inciso I, do Codigo de "
    "Processo Civil, resolvo o merito da demanda. Condeno a parte vencida ao pagamento "
    "das custas e honorarios advocaticios fixados em dez por cento sobre o valor da "
    "condenacao. Publique-se, registre-se e intimem-se."
)

# ---- Conteúdo distintivo por área (vocabulário parecido, fatos diferentes) ----
TESES_INDENIZACAO = [
    "O autor sofreu negativacao indevida de seu nome junto aos orgaos de protecao ao "
    "credito apos quitar integralmente o debito, configurando dano moral in re ipsa.",
    "A instituicao requerida promoveu cobranca de tarifas bancarias nao contratadas, "
    "gerando descontos sucessivos na conta corrente do consumidor ao longo de meses.",
    "Houve falha na prestacao do servico de telefonia, com cobranca por plano diverso "
    "do efetivamente contratado pelo consumidor, em afronta ao Codigo de Defesa do Consumidor.",
    "O voo contratado foi cancelado sem aviso previo, deixando o passageiro sem "
    "assistencia material por mais de doze horas no aeroporto de conexao.",
    "Produto adquirido apresentou vicio oculto dentro do prazo de garantia e o "
    "fornecedor recusou-se a proceder ao reparo ou a substituicao do bem.",
]
TESES_TRABALHISTA = [
    "O reclamante laborou em jornada extraordinaria habitual sem a devida contraprestacao, "
    "fazendo jus ao pagamento das horas extras com o adicional de cinquenta por cento.",
    "A reclamada deixou de recolher os depositos do fundo de garantia durante todo o "
    "periodo do contrato de trabalho, prejudicando o patrimonio do empregado.",
    "O empregado foi dispensado sem justa causa e nao recebeu as verbas rescisorias no "
    "prazo legal, ensejando a aplicacao da multa prevista na legislacao trabalhista.",
]
FATOS_OUTROS = [
    "Trata-se de pedido de revisao de clausulas de contrato de financiamento imobiliario "
    "com alegacao de capitalizacao indevida de juros e cobranca de comissao de permanencia.",
    "Cuida-se de acao de usucapiao extraordinaria de imovel urbano com posse mansa e "
    "pacifica por prazo superior ao exigido em lei, sem oposicao de terceiros.",
    "Demanda relativa a inventario e partilha de bens deixados pelo de cujus, com "
    "discussao acerca da colacao de doacoes realizadas em vida aos herdeiros necessarios.",
]

NOMES = ["Joao da Silva", "Maria Oliveira", "Carlos Pereira", "Ana Souza", "Paulo Costa",
         "Banco Nacional S.A.", "Telecom Brasil Ltda", "Companhia Aerea Azul Celeste",
         "Comercio de Eletronicos Beta", "Construtora Horizonte"]
VALORES = ["R$ 5.000,00", "R$ 12.300,50", "R$ 850,00", "R$ 47.900,00", "R$ 3.200,00",
           "R$ 28.750,00", "R$ 1.150,00", "R$ 99.000,00"]
DATAS = ["12 de marco de 2023", "05 de janeiro de 2024", "30 de junho de 2022",
         "19 de setembro de 2023", "08 de novembro de 2021", "23 de fevereiro de 2024"]
RUAS = ["macieiras", "flores", "acacias", "palmeiras", "jacarandas", "ipes"]
BAIRROS = ["centro", "jardim america", "vila nova", "bela vista", "alto da boa vista"]


def _case_facts(c: int, offset: int = 0) -> list[str]:
    """Sentenças de fato ESPECÍFICAS do caso (tokens únicos por caso).

    São o conteúdo distintivo que uma VERSÃO do mesmo documento preserva e que
    um documento DIFERENTE (ainda que da mesma área) não compartilha.
    `offset` gera fatos de outra matéria (usado no documento 'diferente').
    """
    n = c + offset
    rua = RUAS[n % len(RUAS)]
    bairro = BAIRROS[n % len(BAIRROS)]
    return [
        f"os fatos narrados ocorreram no imovel situado a rua das {rua} numero {100 + n * 3}, "
        f"no bairro {bairro}, onde a parte autora reside ha mais de {2 + n % 9} anos",
        f"consta dos autos o documento de identificacao numero {500000 + n * 137} e o "
        f"comprovante de residencia sob protocolo interno {7000 + n * 11}",
    ]


@dataclass
class Sample:
    doc_id: str
    case_id: str
    text: str
    base_doc_id: str  # com qual base ele forma um par
    label: str        # copia | versao | diferente


def _fill(thesis: str, facts: list[str], rnd: random.Random):
    parte_a, parte_b = rnd.sample(NOMES, 2)
    valor = rnd.choice(VALORES)
    data = rnd.choice(DATAS)
    fatos = (
        f"DOS FATOS. Em {data}, a parte autora {parte_a} foi vitima dos fatos a seguir. "
        f"{thesis} {facts[0]}. {facts[1]}. "
        f"A parte requerida {parte_b} deu causa ao dano, estimado em {valor}. "
        f"O nexo causal entre a conduta e o prejuizo restou cabalmente demonstrado."
    )
    return fatos, dict(valor=valor, data=data, autor=parte_a, reu=parte_b)


def _ocr_noise(text: str, rnd: random.Random) -> str:
    """Simula ruído leve de OCR/digitalização (não muda o conteúdo)."""
    out = []
    for ch in text:
        r = rnd.random()
        if r < 0.01 and ch == " ":
            out.append("  ")          # espaço duplo (removido na normalização)
        elif r < 0.004 and ch.isalpha():
            out.append(ch + ch)        # ruído leve de OCR: caractere duplicado esporádico
        else:
            out.append(ch)
    noisy = "".join(out)
    return "Folha 3\n" + noisy + "\npag. 3/8"  # cabeçalho/rodapé que a normalização remove


def _petition(thesis: str, facts: list[str], rnd: random.Random):
    fatos, meta = _fill(thesis, facts, rnd)
    text = " ".join([PETICAO_HEADER, fatos, PETICAO_DIREITO, PETICAO_PEDIDO])
    return text, meta


def _make_version(thesis: str, facts: list[str], meta: dict, rnd: random.Random) -> str:
    """Versão modificada: MESMO caso (preserva o 1º fato específico e a tese),
    mas troca partes/valores/datas e REESCREVE o 2º fato — edições localizadas."""
    novo_a, novo_b = rnd.sample([n for n in NOMES if n not in (meta["autor"], meta["reu"])], 2)
    novo_valor = rnd.choice([v for v in VALORES if v != meta["valor"]])
    nova_data = rnd.choice([d for d in DATAS if d != meta["data"]])
    fato2_reescrito = ("a documentacao pertinente ao caso foi devidamente juntada aos autos "
                       "pela parte interessada no momento processual oportuno")
    fatos = (
        f"DOS FATOS. Em {nova_data}, a parte autora {novo_a} foi vitima dos fatos a seguir. "
        f"{thesis} {facts[0]}. {fato2_reescrito}. "
        f"A parte requerida {novo_b} deu causa ao dano, estimado em {novo_valor}. "
        f"O nexo causal entre a conduta e o prejuizo restou cabalmente demonstrado."
    )
    return " ".join([PETICAO_HEADER, fatos, PETICAO_DIREITO, PETICAO_PEDIDO])


def build_corpus(num_cases: int = 40, seed: int = 7) -> tuple[list[Sample], dict]:
    rnd = random.Random(seed)
    samples: list[Sample] = []
    for c in range(num_cases):
        case_id = f"case_{c:03d}"
        # escolhe área do caso (define o vocabulário compartilhado entre base e a "armadilha")
        area_pool = rnd.choice([TESES_INDENIZACAO, TESES_TRABALHISTA])
        thesis = rnd.choice(area_pool)
        facts = _case_facts(c)
        base_text, meta = _petition(thesis, facts, rnd)
        base_id = f"{case_id}_base"
        samples.append(Sample(base_id, case_id, base_text, base_id, "copia"))  # auto-par ignorado depois

        # 1) cópia exata
        samples.append(Sample(f"{case_id}_copy", case_id, base_text, base_id, "copia"))
        # 2) cópia com ruído de OCR
        samples.append(Sample(f"{case_id}_ocr", case_id, _ocr_noise(base_text, rnd), base_id, "copia"))
        # 3) versão (mesmo caso: preserva fatos específicos, edições localizadas)
        samples.append(Sample(f"{case_id}_ver", case_id, _make_version(thesis, facts, meta, rnd), base_id, "versao"))
        # 4) DIFERENTE mas mesma área: MESMO boilerplate + MESMO vocabulário,
        #    outra tese E outros fatos específicos (a "armadilha do boilerplate")
        other_thesis = rnd.choice([t for t in area_pool if t != thesis])
        diff_area_text, _ = _petition(other_thesis, _case_facts(c, offset=1000), rnd)
        samples.append(Sample(f"{case_id}_diffarea", case_id, diff_area_text, base_id, "diferente"))
        # 5) DIFERENTE de outro assunto
        outro = rnd.choice(FATOS_OUTROS)
        outro_text = " ".join([SENTENCA_HEADER, outro, SENTENCA_DISPOSITIVO])
        samples.append(Sample(f"{case_id}_other", case_id, outro_text, base_id, "diferente"))

    meta = dict(num_cases=num_cases, seed=seed, total_docs=len(samples))
    return samples, meta


def labeled_pairs(samples: list[Sample]) -> list[tuple[str, str, str]]:
    """Pares (base_id, variante_id, label) — só dentro do mesmo caso."""
    bases = {s.case_id: s.doc_id for s in samples if s.doc_id.endswith("_base")}
    pairs = []
    for s in samples:
        if s.doc_id.endswith("_base"):
            continue
        pairs.append((bases[s.case_id], s.doc_id, s.label))
    return pairs
