"""
Test script for semantic chunking implementation with realistic long-form content.
Includes text, image summaries, chart summaries, and table summaries.
"""
import sys
from pathlib import Path

# Add project root to path for imports
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# Test 1: Basic imports
print("=" * 60)
print("TEST 1: Import checks")
print("=" * 60)

try:
    from chunking.chunking_utils import (
        SemanticChunker,
        cosine_similarity,
        calculate_sentence_similarities,
        find_semantic_breakpoints,
        create_semantic_multimodal_chunks,
        sentence_split,
        deterministic_chunk_id
    )
    import numpy as np
    print("✓ All imports successful")
except ImportError as e:
    print(f"✗ Import failed: {e}")
    sys.exit(1)

# =============================================================================
# REALISTIC LONG-FORM CONTENT FOR TESTING
# =============================================================================

# Long text simulating a research paper about climate change
CLIMATE_RESEARCH_TEXT = """
Climate change represents one of the most significant challenges facing humanity in the 21st century. Global temperatures have risen approximately 1.1 degrees Celsius above pre-industrial levels, with the past decade being the warmest on record. Scientific consensus, supported by extensive research from institutions worldwide, confirms that human activities, particularly the burning of fossil fuels, are the primary drivers of this warming trend.

The Intergovernmental Panel on Climate Change has published comprehensive reports detailing the observed impacts and projected future scenarios. These reports synthesize findings from thousands of peer-reviewed studies and represent the collective understanding of climate scientists globally. The evidence indicates that without significant reductions in greenhouse gas emissions, global temperatures could rise by 2.5 to 4.5 degrees Celsius by the end of this century.

Ocean systems are experiencing profound changes as a result of climate change. Sea surface temperatures have increased, leading to coral bleaching events that threaten marine biodiversity. Ocean acidification, caused by the absorption of excess carbon dioxide, poses additional risks to shell-forming organisms and the broader marine food web. Rising sea levels, driven by thermal expansion and melting ice sheets, threaten coastal communities worldwide.

The Arctic region is warming at nearly twice the global average rate, a phenomenon known as Arctic amplification. This accelerated warming is causing dramatic reductions in sea ice extent, with summer ice coverage declining by approximately 13 percent per decade. The loss of reflective ice surfaces creates a feedback loop, as darker ocean waters absorb more solar radiation, further accelerating warming.

Extreme weather events are becoming more frequent and intense due to climate change. Heat waves, droughts, and wildfires have increased in severity across multiple continents. Precipitation patterns are shifting, with some regions experiencing more intense rainfall and flooding while others face prolonged dry periods. These changes have significant implications for agriculture, water resources, and human health.

The economic impacts of climate change are substantial and growing. Estimates suggest that unmitigated climate change could reduce global GDP by 10 to 23 percent by 2100. The costs of climate-related disasters have already increased dramatically, with insured losses from weather events reaching record levels in recent years. Developing nations, despite contributing least to historical emissions, often face the greatest vulnerability to climate impacts.

Mitigation strategies focus on reducing greenhouse gas emissions through the transition to renewable energy sources, improved energy efficiency, and changes in land use practices. Solar and wind power have experienced dramatic cost reductions, making them increasingly competitive with fossil fuels. Electric vehicles are gaining market share, and many countries have announced plans to phase out internal combustion engines in the coming decades.

Adaptation measures are equally important, as some degree of climate change is now unavoidable due to historical emissions. These measures include developing drought-resistant crops, improving water management systems, strengthening infrastructure to withstand extreme weather, and implementing early warning systems for climate-related disasters. Coastal communities are exploring options such as sea walls, managed retreat, and nature-based solutions like wetland restoration.

International cooperation is essential for addressing climate change effectively. The Paris Agreement, adopted in 2015, established a framework for global action, with countries committing to limit warming to well below 2 degrees Celsius. However, current national commitments fall short of what is needed to achieve this goal, highlighting the need for increased ambition and accelerated implementation of climate policies.
"""

# Image summaries (simulating vision model output)
IMAGE_SUMMARIES = {
    "global_temp_map": """
This image shows a global temperature anomaly map for the year 2023. The map uses a color gradient from blue (cooler than average) to red (warmer than average) to display temperature deviations from the 1951-1980 baseline. Most land masses and ocean regions appear in orange to dark red, indicating temperatures 1-3 degrees Celsius above the historical average. The Arctic region shows the most intense warming, with anomalies exceeding 4 degrees Celsius in some areas. Notable hotspots include Siberia, northern Canada, and parts of the Antarctic Peninsula. The visualization clearly demonstrates the global nature of warming, with very few regions showing below-average temperatures.
""",
    "ice_extent_comparison": """
This satellite image comparison shows Arctic sea ice extent in September 1980 versus September 2023. The left panel displays the 1980 ice coverage, showing a continuous ice sheet extending across most of the Arctic Ocean and connecting to the coastlines of Russia, Canada, and Greenland. The right panel shows the 2023 ice extent, which is dramatically reduced, with open water visible throughout much of the Arctic basin. The Northwest Passage appears largely ice-free, and the ice edge has retreated hundreds of kilometers from historical positions. The median ice edge from 1981-2010 is overlaid as a reference line, highlighting the substantial loss of ice coverage over the past four decades.
""",
    "renewable_energy_installation": """
This photograph shows a large-scale solar farm in the Mojave Desert, California. Thousands of photovoltaic panels are arranged in neat rows stretching toward the horizon, covering approximately 3,500 acres of desert terrain. The panels are mounted on tracking systems that follow the sun throughout the day to maximize energy generation. In the background, transmission towers carry electricity to the grid. The facility has a nameplate capacity of 550 megawatts, enough to power approximately 180,000 homes. The image illustrates the scale of modern renewable energy infrastructure and the potential for solar power in high-insolation regions.
""",
    "flood_damage_assessment": """
This aerial photograph documents flooding in Pakistan during the 2022 monsoon season. The image shows a vast expanse of floodwater covering agricultural land and submerging villages in Sindh Province. Roads and infrastructure are barely visible above the waterline, and displaced residents can be seen gathered on elevated areas and rooftops. The brown color of the floodwater indicates high sediment content from erosion. Emergency response boats are visible conducting rescue operations. This flooding affected over 33 million people and caused an estimated 30 billion dollars in damages, exemplifying the devastating human cost of climate-intensified extreme weather events.
"""
}

# Chart summaries (simulating vision model output)
CHART_SUMMARIES = {
    "temperature_trend": """
This line chart displays global mean surface temperature anomalies from 1880 to 2023, relative to the 1951-1980 average. The x-axis shows years, and the y-axis shows temperature anomaly in degrees Celsius, ranging from -0.5 to +1.5. The chart includes both annual values (shown as gray bars) and a 5-year moving average (shown as a bold red line). The trend shows relatively stable temperatures from 1880 to about 1910, followed by gradual warming through the mid-20th century. A notable cooling period occurred from roughly 1940-1970, attributed to aerosol emissions. Since 1970, warming has accelerated dramatically, with the rate increasing from 0.1 degrees per decade to 0.2 degrees per decade in recent years. The years 2016, 2020, and 2023 appear as the three warmest on record, each exceeding 1.0 degrees above the baseline.
""",
    "emissions_by_sector": """
This pie chart breaks down global greenhouse gas emissions by economic sector for the year 2022. The largest contributor is Energy (electricity and heat production) at 25%, followed by Agriculture, Forestry and Land Use at 24%, Industry at 21%, Transportation at 16%, Buildings at 6%, and Other at 8%. A secondary breakdown shows that fossil fuel combustion accounts for 73% of total emissions, while land use changes contribute 18% and industrial processes contribute 9%. The chart uses distinct colors for each sector and includes both percentage values and absolute emissions in gigatons of CO2 equivalent. A note indicates that transportation emissions have grown fastest over the past two decades, while energy sector emissions have begun to plateau in some regions due to renewable energy deployment.
""",
    "sea_level_projection": """
This chart shows observed and projected global mean sea level rise from 1900 to 2150. Historical observations from tide gauges and satellite altimetry are shown as a black line from 1900 to 2023, indicating approximately 20 centimeters of rise over this period. Future projections are shown as colored bands representing different emissions scenarios. The low emissions scenario (SSP1-2.6) projects 30-60 cm of rise by 2100, shown in blue. The intermediate scenario (SSP2-4.5) projects 45-80 cm, shown in yellow. The high emissions scenario (SSP5-8.5) projects 60-110 cm, shown in red. All scenarios show continued rise beyond 2100, with the high emissions pathway potentially reaching 2 meters by 2150. Uncertainty ranges widen over time, reflecting unknowns about ice sheet dynamics.
""",
    "renewable_cost_decline": """
This dual-axis chart tracks the levelized cost of electricity (LCOE) for solar PV and onshore wind alongside their cumulative installed capacity from 2010 to 2023. The left y-axis shows LCOE in dollars per megawatt-hour, ranging from 0 to 400. The right y-axis shows cumulative capacity in gigawatts. Solar PV costs (orange line) declined from $378/MWh in 2010 to $44/MWh in 2023, an 88% reduction. Wind costs (blue line) fell from $95/MWh to $42/MWh, a 56% reduction. Meanwhile, solar capacity (orange area) grew from 40 GW to 1,400 GW, and wind capacity (blue area) grew from 198 GW to 940 GW. The chart demonstrates the learning curve effect, where costs decrease as deployment increases, and shows that renewables are now cost-competitive with fossil fuels in most markets.
"""
}

# Table summaries (simulating vision model output)
TABLE_SUMMARIES = {
    "emissions_by_country": """
This table presents the top 10 greenhouse gas emitting countries for 2022, with columns for Country, Total Emissions (GT CO2e), Percentage of Global Total, Per Capita Emissions (tons), and Change from 2010. China leads with 14.3 GT (27%), followed by the United States at 5.2 GT (10%), India at 3.9 GT (7%), and the European Union at 3.3 GT (6%). Russia, Japan, Brazil, Indonesia, Iran, and Canada complete the top 10. Per capita emissions vary significantly, with the US highest at 15.5 tons per person, while India is lowest at 2.8 tons. The change column shows China increased 35% since 2010, India increased 45%, while the US decreased 8% and the EU decreased 22%. A footer note indicates that historical cumulative emissions show different rankings, with the US responsible for 25% of all emissions since 1850.
""",
    "climate_impacts_by_region": """
This table summarizes projected climate impacts by world region under a 2-degree warming scenario. Columns include Region, Primary Risks, Population Exposed (millions), Economic Impact (% GDP), and Adaptation Priority. Africa faces water scarcity and crop failure, affecting 350 million people with 4.7% GDP impact. South Asia shows extreme heat and flooding risks for 800 million people and 5.2% GDP impact. Southeast Asia faces sea level rise and typhoon intensification, threatening 250 million with 4.1% GDP impact. Small Island States face existential threats from sea level rise, affecting 65 million with 15%+ GDP impact. The table highlights that developing regions face disproportionate impacts despite lower historical emissions, and notes that adaptation costs could reach $300 billion annually by 2030.
""",
    "renewable_targets_by_country": """
This table compares renewable energy targets and progress for major economies. Columns show Country, 2030 Target (% of electricity), Current Status (2023), Policy Mechanisms, and Investment (billion USD in 2022). Germany targets 80% renewable electricity, currently at 52%, using feed-in tariffs and auctions, with $36B invested. China targets 50%, currently at 32%, using mandates and subsidies, leading global investment at $546B. The United States targets 40%, currently at 22%, using tax credits and state mandates, with $141B invested. India targets 50%, currently at 20%, using auctions and manufacturing incentives, with $28B invested. The UK, Japan, and Australia are also included with their respective metrics. A summary row indicates global renewable investment reached $1.1 trillion in 2022, exceeding fossil fuel investment for the first time.
""",
    "extreme_weather_statistics": """
This table presents statistics on extreme weather events from 2013-2023 compared to 1983-1993. Categories include Heat Waves, Severe Floods, Major Droughts, Category 4-5 Hurricanes, and Wildfires over 100,000 acres. For each category, columns show Count (historical vs recent decade), Percentage Change, Average Duration Change, and Economic Losses. Heat waves increased from 45 to 98 events (+118%), with duration increasing 4 days on average. Severe floods rose from 127 to 223 events (+76%). Major droughts increased from 15 to 28 (+87%), with average duration extending by 30%. Category 4-5 hurricanes remained stable in count but increased in intensity. Large wildfires increased from 52 to 121 (+133%). Total economic losses from these events grew from $890 billion to $2.8 trillion across the two decades. Attribution science indicates climate change increased the likelihood of most events by 2-10 times.
"""
}

# Test 2: Sentence splitting with long text
print("\n" + "=" * 60)
print("TEST 2: Sentence splitting with long climate text")
print("=" * 60)

sentences = sentence_split(CLIMATE_RESEARCH_TEXT)
print(f"Input text length: {len(CLIMATE_RESEARCH_TEXT)} characters")
print(f"Total sentences found: {len(sentences)}")
print(f"First 3 sentences:")
for i, s in enumerate(sentences[:3]):
    print(f"  [{i}] {s[:80]}...")
print(f"Last 3 sentences:")
for i, s in enumerate(sentences[-3:], len(sentences)-3):
    print(f"  [{i}] {s[:80]}...")

assert len(sentences) > 20, f"Expected many sentences, got {len(sentences)}"
print("✓ Sentence splitting works correctly with long text")

# Test 3: Cosine similarity
print("\n" + "=" * 60)
print("TEST 3: Cosine similarity calculation")
print("=" * 60)

vec1 = np.array([1.0, 0.0, 0.0])
vec2 = np.array([1.0, 0.0, 0.0])
vec3 = np.array([0.0, 1.0, 0.0])

sim_same = cosine_similarity(vec1, vec2)
sim_orthogonal = cosine_similarity(vec1, vec3)

print(f"Same vectors: {sim_same:.4f} (expected: 1.0)")
print(f"Orthogonal vectors: {sim_orthogonal:.4f} (expected: 0.0)")

assert abs(sim_same - 1.0) < 0.001, "Same vectors should have similarity 1.0"
assert abs(sim_orthogonal - 0.0) < 0.001, "Orthogonal vectors should have similarity 0.0"
print("✓ Cosine similarity calculation works correctly")

# Test 4: Breakpoint detection
print("\n" + "=" * 60)
print("TEST 4: Semantic breakpoint detection")
print("=" * 60)

# Similarities simulating topic shifts in the climate text
test_similarities = [0.92, 0.88, 0.85, 0.35, 0.91, 0.89, 0.28, 0.87, 0.90, 0.32, 0.88]
# Topic shifts at indices 3, 6, and 9

breakpoints = find_semantic_breakpoints(test_similarities, threshold=0.5)
print(f"Similarities: {test_similarities}")
print(f"Detected breakpoints: {breakpoints}")

assert 3 in breakpoints and 6 in breakpoints and 9 in breakpoints
print("✓ Breakpoint detection correctly identifies topic shifts")

# Test 5: SemanticChunker with mock embedding client on climate text
print("\n" + "=" * 60)
print("TEST 5: Semantic chunking of climate research text")
print("=" * 60)

class ClimateEmbeddingClient:
    """Mock embedding client that simulates topic-aware embeddings for climate text."""

    def embed(self, texts):
        embeddings = []
        for text in texts:
            text_lower = text.lower()
            # Create topic-specific embeddings
            if any(w in text_lower for w in ['temperature', 'warming', 'degrees', 'celsius', 'heat']):
                emb = [0.9, 0.1, 0.0, 0.0, 0.0]  # Temperature topic
            elif any(w in text_lower for w in ['ocean', 'sea', 'marine', 'coral', 'coastal']):
                emb = [0.1, 0.9, 0.0, 0.0, 0.0]  # Ocean topic
            elif any(w in text_lower for w in ['arctic', 'ice', 'polar', 'glacier']):
                emb = [0.0, 0.1, 0.9, 0.0, 0.0]  # Arctic topic
            elif any(w in text_lower for w in ['extreme', 'weather', 'drought', 'flood', 'wildfire']):
                emb = [0.0, 0.0, 0.1, 0.9, 0.0]  # Extreme weather topic
            elif any(w in text_lower for w in ['economic', 'cost', 'gdp', 'billion', 'dollar']):
                emb = [0.0, 0.0, 0.0, 0.1, 0.9]  # Economic topic
            elif any(w in text_lower for w in ['renewable', 'solar', 'wind', 'energy', 'mitigation']):
                emb = [0.5, 0.0, 0.0, 0.0, 0.5]  # Mitigation topic
            elif any(w in text_lower for w in ['adaptation', 'resilience', 'infrastructure']):
                emb = [0.0, 0.5, 0.0, 0.5, 0.0]  # Adaptation topic
            else:
                emb = [0.3, 0.3, 0.2, 0.1, 0.1]  # General
            embeddings.append(emb)
        return embeddings

climate_client = ClimateEmbeddingClient()

chunker = SemanticChunker(
    embedding_client=climate_client,
    similarity_threshold=0.6,
    percentile_threshold=None,
    min_chunk_size=30,
    max_chunk_size=200
)

climate_chunks = chunker.chunk_text(
    text=CLIMATE_RESEARCH_TEXT,
    pdf_name="climate_report.pdf",
    page_no=1,
    start_chunk_number=0
)

print(f"Input: Climate research text ({len(CLIMATE_RESEARCH_TEXT)} chars)")
print(f"Chunks created: {len(climate_chunks)}")
print("\nChunk breakdown:")
for i, chunk in enumerate(climate_chunks):
    word_count = len(chunk['text'].split())
    preview = chunk['text'][:100].replace('\n', ' ')
    method = chunk.get('metadata', {}).get('chunking_method', 'unknown')
    print(f"  Chunk {i}: {word_count} words ({method})")
    print(f"    Preview: {preview}...")
    print()

assert len(climate_chunks) >= 3, "Should create multiple semantic chunks"
print("✓ Semantic chunking of long text works correctly")

# Test 6: Multimodal chunking with images, charts, and tables
print("\n" + "=" * 60)
print("TEST 6: Multimodal semantic chunking (text + images + charts + tables)")
print("=" * 60)

# Create realistic multimodal content blocks
multimodal_blocks = [
    # Introduction text
    {
        "type": "text",
        "content": "Climate change represents one of the most significant challenges facing humanity in the 21st century. Global temperatures have risen approximately 1.1 degrees Celsius above pre-industrial levels, with the past decade being the warmest on record. Scientific consensus, supported by extensive research from institutions worldwide, confirms that human activities are the primary drivers of this warming trend.",
        "position": 0
    },
    # Temperature map image
    {
        "type": "image",
        "content": IMAGE_SUMMARIES["global_temp_map"].strip(),
        "image_link": "/outputs/batch123/climate_report/images/global_temp_map.png",
        "position": 1
    },
    # Temperature trend chart
    {
        "type": "table",
        "content": CHART_SUMMARIES["temperature_trend"].strip(),
        "table_link": "/outputs/batch123/climate_report/tables/temperature_trend_chart.png",
        "position": 2
    },
    # Ocean impacts text
    {
        "type": "text",
        "content": "Ocean systems are experiencing profound changes as a result of climate change. Sea surface temperatures have increased, leading to coral bleaching events that threaten marine biodiversity. Ocean acidification poses additional risks to shell-forming organisms. Rising sea levels threaten coastal communities worldwide with increasing flood risks and erosion.",
        "position": 3
    },
    # Sea level projection chart
    {
        "type": "table",
        "content": CHART_SUMMARIES["sea_level_projection"].strip(),
        "table_link": "/outputs/batch123/climate_report/tables/sea_level_projection.png",
        "position": 4
    },
    # Arctic section text
    {
        "type": "text",
        "content": "The Arctic region is warming at nearly twice the global average rate, a phenomenon known as Arctic amplification. This accelerated warming is causing dramatic reductions in sea ice extent, with summer ice coverage declining by approximately 13 percent per decade. The loss of reflective ice surfaces creates a feedback loop that further accelerates warming.",
        "position": 5
    },
    # Ice comparison image
    {
        "type": "image",
        "content": IMAGE_SUMMARIES["ice_extent_comparison"].strip(),
        "image_link": "/outputs/batch123/climate_report/images/ice_comparison.png",
        "position": 6
    },
    # Extreme weather text
    {
        "type": "text",
        "content": "Extreme weather events are becoming more frequent and intense due to climate change. Heat waves, droughts, and wildfires have increased in severity across multiple continents. Precipitation patterns are shifting, with some regions experiencing more intense rainfall while others face prolonged dry periods.",
        "position": 7
    },
    # Extreme weather statistics table
    {
        "type": "table",
        "content": TABLE_SUMMARIES["extreme_weather_statistics"].strip(),
        "table_link": "/outputs/batch123/climate_report/tables/extreme_weather_stats.png",
        "position": 8
    },
    # Flood damage image
    {
        "type": "image",
        "content": IMAGE_SUMMARIES["flood_damage_assessment"].strip(),
        "image_link": "/outputs/batch123/climate_report/images/pakistan_floods.png",
        "position": 9
    },
    # Economic impacts text
    {
        "type": "text",
        "content": "The economic impacts of climate change are substantial and growing. Estimates suggest that unmitigated climate change could reduce global GDP by 10 to 23 percent by 2100. The costs of climate-related disasters have already increased dramatically, with insured losses reaching record levels in recent years.",
        "position": 10
    },
    # Emissions by country table
    {
        "type": "table",
        "content": TABLE_SUMMARIES["emissions_by_country"].strip(),
        "table_link": "/outputs/batch123/climate_report/tables/emissions_by_country.png",
        "position": 11
    },
    # Emissions by sector chart
    {
        "type": "table",
        "content": CHART_SUMMARIES["emissions_by_sector"].strip(),
        "table_link": "/outputs/batch123/climate_report/tables/emissions_by_sector.png",
        "position": 12
    },
    # Mitigation text
    {
        "type": "text",
        "content": "Mitigation strategies focus on reducing greenhouse gas emissions through the transition to renewable energy sources, improved energy efficiency, and changes in land use practices. Solar and wind power have experienced dramatic cost reductions, making them increasingly competitive with fossil fuels.",
        "position": 13
    },
    # Renewable energy image
    {
        "type": "image",
        "content": IMAGE_SUMMARIES["renewable_energy_installation"].strip(),
        "image_link": "/outputs/batch123/climate_report/images/solar_farm.png",
        "position": 14
    },
    # Renewable cost chart
    {
        "type": "table",
        "content": CHART_SUMMARIES["renewable_cost_decline"].strip(),
        "table_link": "/outputs/batch123/climate_report/tables/renewable_costs.png",
        "position": 15
    },
    # Renewable targets table
    {
        "type": "table",
        "content": TABLE_SUMMARIES["renewable_targets_by_country"].strip(),
        "table_link": "/outputs/batch123/climate_report/tables/renewable_targets.png",
        "position": 16
    },
    # Conclusion text
    {
        "type": "text",
        "content": "International cooperation is essential for addressing climate change effectively. The Paris Agreement established a framework for global action, with countries committing to limit warming to well below 2 degrees Celsius. However, current national commitments fall short of what is needed, highlighting the need for increased ambition.",
        "position": 17
    },
    # Regional impacts table
    {
        "type": "table",
        "content": TABLE_SUMMARIES["climate_impacts_by_region"].strip(),
        "table_link": "/outputs/batch123/climate_report/tables/regional_impacts.png",
        "position": 18
    }
]

# Perform multimodal semantic chunking
multimodal_chunks = create_semantic_multimodal_chunks(
    blocks=multimodal_blocks,
    pdf_name="climate_report.pdf",
    page_no=1,
    embedding_client=climate_client,
    similarity_threshold=0.6,
    percentile_threshold=None,
    min_chunk_size=20,
    max_chunk_size=300,
    start_chunk_number=0
)

print(f"Input: {len(multimodal_blocks)} multimodal blocks")
print(f"  - Text blocks: {sum(1 for b in multimodal_blocks if b['type'] == 'text')}")
print(f"  - Image blocks: {sum(1 for b in multimodal_blocks if b['type'] == 'image')}")
print(f"  - Table/Chart blocks: {sum(1 for b in multimodal_blocks if b['type'] == 'table')}")
print(f"\nChunks created: {len(multimodal_chunks)}")

# Count by content type
content_types = {}
for chunk in multimodal_chunks:
    ct = chunk.get('content_type', 'text')
    content_types[ct] = content_types.get(ct, 0) + 1

print(f"\nChunk breakdown by type:")
for ct, count in content_types.items():
    print(f"  - {ct}: {count} chunks")

print(f"\nDetailed chunk list:")
for i, chunk in enumerate(multimodal_chunks):
    ct = chunk.get('content_type', 'text')
    word_count = len(chunk['text'].split())
    has_image = bool(chunk.get('image_link'))
    has_table = bool(chunk.get('table_link'))

    link_info = ""
    if has_image:
        link_info = f" [IMG: {chunk['image_link'].split('/')[-1]}]"
    elif has_table:
        link_info = f" [TBL: {chunk['table_link'].split('/')[-1]}]"

    preview = chunk['text'][:60].replace('\n', ' ')
    print(f"  [{i:2d}] {ct:6s} | {word_count:3d} words{link_info}")
    print(f"       Preview: {preview}...")

assert content_types.get('text', 0) > 0, "Should have text chunks"
assert content_types.get('image', 0) > 0, "Should have image chunks"
assert content_types.get('table', 0) > 0, "Should have table chunks"
print("\n✓ Multimodal semantic chunking works correctly")

# Test 7: Context linking verification
print("\n" + "=" * 60)
print("TEST 7: Context linking between chunks")
print("=" * 60)

print("Verifying context chain:")
for i, chunk in enumerate(multimodal_chunks):
    before_id = chunk.get('context_before_id', '')
    after_id = chunk.get('context_after_id', '')

    # Verify first chunk has no before
    if i == 0:
        assert before_id == '', "First chunk should have no before context"
    else:
        assert before_id == multimodal_chunks[i-1]['chunk_id'], f"Chunk {i} before_id mismatch"

    # Verify last chunk has no after
    if i == len(multimodal_chunks) - 1:
        assert after_id == '', "Last chunk should have no after context"
    else:
        assert after_id == multimodal_chunks[i+1]['chunk_id'], f"Chunk {i} after_id mismatch"

print(f"  ✓ All {len(multimodal_chunks)} chunks correctly linked")
print("✓ Context linking works correctly")

# Test 8: Chunk ID determinism
print("\n" + "=" * 60)
print("TEST 8: Deterministic chunk IDs")
print("=" * 60)

# Run chunking again with same input
multimodal_chunks_2 = create_semantic_multimodal_chunks(
    blocks=multimodal_blocks,
    pdf_name="climate_report.pdf",
    page_no=1,
    embedding_client=climate_client,
    similarity_threshold=0.6,
    percentile_threshold=None,
    min_chunk_size=20,
    max_chunk_size=300,
    start_chunk_number=0
)

assert len(multimodal_chunks) == len(multimodal_chunks_2), "Should produce same number of chunks"

matching_ids = sum(1 for c1, c2 in zip(multimodal_chunks, multimodal_chunks_2)
                   if c1['chunk_id'] == c2['chunk_id'])
print(f"  Matching chunk IDs: {matching_ids}/{len(multimodal_chunks)}")

assert matching_ids == len(multimodal_chunks), "All chunk IDs should match"
print("✓ Deterministic chunk IDs work correctly")

# Test 9: Fallback mode (no embedding client)
print("\n" + "=" * 60)
print("TEST 9: Fallback mode without embedding client")
print("=" * 60)

fallback_chunks = create_semantic_multimodal_chunks(
    blocks=multimodal_blocks[:5],  # Use subset for faster test
    pdf_name="climate_report.pdf",
    page_no=1,
    embedding_client=None,  # No embedding client
    min_chunk_size=20,
    max_chunk_size=150,
    start_chunk_number=0
)

print(f"Chunks created in fallback mode: {len(fallback_chunks)}")
fallback_methods = set()
for chunk in fallback_chunks:
    method = chunk.get('metadata', {}).get('chunking_method', 'none')
    fallback_methods.add(method)

print(f"Chunking methods used: {fallback_methods}")
assert len(fallback_chunks) > 0, "Should create chunks even without embedding client"
print("✓ Fallback mode works correctly")

# Summary
print("\n" + "=" * 60)
print("ALL TESTS PASSED ✓")
print("=" * 60)
print("""
Summary of tested functionality:
1. Import checks - All modules load correctly
2. Sentence splitting - Works with long-form text
3. Cosine similarity - Accurate vector calculations
4. Breakpoint detection - Identifies topic shifts
5. Semantic chunking - Topic-aware text segmentation
6. Multimodal chunking - Handles text, images, charts, tables
7. Context linking - Chunks properly linked to neighbors
8. Deterministic IDs - Reproducible chunk identifiers
9. Fallback mode - Works without embedding client

Content tested:
- Long climate research text (~5000 chars)
- 4 image summaries (temperature map, ice extent, solar farm, flood damage)
- 4 chart summaries (temperature trend, emissions, sea level, renewable costs)
- 4 table summaries (emissions by country, regional impacts, renewable targets, weather stats)
- Full multimodal document with 19 blocks
""")
