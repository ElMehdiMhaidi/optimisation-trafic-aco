import streamlit as st
import networkx as nx
import matplotlib.pyplot as plt
import random
import numpy as np
import math
from PIL import Image, ImageDraw, ImageFont
import os
import platform
import datetime
import streamlit.components.v1 as components

def inject_disable_scroll_js():
    components.html(
        """
    <script>
    const sliders = document.querySelectorAll('div[data-baseweb="slider"]');
    sliders.forEach(slider => {
      let isDragging = false;

      slider.addEventListener('mousedown', () => {
        isDragging = true;
        document.body.style.overflow = 'hidden'; // Disable scroll
      });

      slider.addEventListener('mouseup', () => {
        isDragging = false;
        document.body.style.overflow = 'auto';   // Enable scroll
      });

      slider.addEventListener('mouseleave', () => {  // Handle cases where mouse leaves slider while dragging
        if (isDragging) {
          isDragging = false;
          document.body.style.overflow = 'auto';   // Enable scroll
        }
      });
    });

    // Touch events (for mobile)
    sliders.forEach(slider => {
      slider.addEventListener('touchstart', () => {
        document.body.style.overflow = 'hidden'; // Disable scroll
      });

      slider.addEventListener('touchend', () => {
        document.body.style.overflow = 'auto';   // Enable scroll
      });

      slider.addEventListener('touchcancel', () => {  // Optional: Handle touch cancel events
        document.body.style.overflow = 'auto';   // Enable scroll
      });

      slider.addEventListener('touchmove', (event) => {
        // Prevent default touchmove behavior within the slider area
        event.stopPropagation();
      }, { passive: false }); // Use passive: false to allow preventDefault()
    });
    </script>
    """,
        height=0,
        width=0,
    )

inject_disable_scroll_js()  # Call the function to inject the JavaScript
st.title("My Streamlit App")

# Example Slider
value = st.slider("Slider Example", 0, 100, 50)
st.write("Slider Value:", value)

# --- Constantes de trafic spécifiques aux villes ---
city_traffic_params = {
    "Los Angeles 🇺🇸 (Fort Trafic)": {
        "rho_max_global": 400,
        "congestion_factor_morning_peak": 2.0,
        "congestion_factor_lunch_peak": 1.5,
        "congestion_factor_evening_peak": 2.5,
        "v_max": 15,
        "mu_T": 10800,
        "sigma_T": 9000,
    },
    "Lyon 🇫🇷 (Trafic Moyen)": {
        "rho_max_global": 300,
        "congestion_factor_morning_peak": 1.5,
        "congestion_factor_lunch_peak": 1,
        "congestion_factor_evening_peak": 1.5,
        "v_max": 16.66,
        "mu_T": 14400,
        "sigma_T": 7200,
    },
    "Ben Guerir 🇲🇦 (Faible Trafic)": {
        "rho_max_global": 100,
        "congestion_factor_morning_peak": 1.2,
        "congestion_factor_lunch_peak": 0.8,
        "congestion_factor_evening_peak": 0.7,
        "v_max": 16.66,
        "mu_T": 18000,
        "sigma_T": 5400,
    },
    "Personnalisé": {
        "rho_max_global": 300,
        "congestion_factor_morning_peak": 1.5,
        "congestion_factor_lunch_peak": 1.2,
        "congestion_factor_evening_peak": 1.8,
        "v_max": 16.66,
        "mu_T": 14400,
        "sigma_T": 7200,
    }
}

# --- Fonctions de génération de graphe et visualisation ---
def generer_matrice_uniforme(taille, borne_min, borne_max):
    """Génère une matrice aléatoire suivant une loi uniforme."""
    if borne_min >= borne_max:
        raise ValueError("La borne minimale doit être strictement inférieure à la borne maximale.")
    matrice = np.random.uniform(low=borne_min, high=borne_max, size=taille)
    return matrice

def visualize_road_network_pillow(G, start_point, sources, background_image_path="mapgta.jpg", output_image_path="streamlit_graph_aco_path.png", best_path=None):
    """Visualise un graphe de réseau routier avec Pillow."""
    try:
        background_image = Image.open(background_image_path).convert("RGB")
    except FileNotFoundError:
        print(f"Warning: Image '{background_image_path}' not found. Using white background.")
        background_image = Image.new("RGB", (1000, 800), "white")

    draw = ImageDraw.Draw(background_image)
    pos = nx.spring_layout(G, seed=42)

    min_x = min(x for x, y in pos.values())
    max_x = max(x for x, y in pos.values())
    min_y = min(y for x, y in pos.values())
    max_y = max(y for x, y in pos.values())

    graph_width = max_x - min_x
    graph_height = max_y - min_y

    scale_factor = 1.0
    margin = 0.1
    if graph_width > 0 and graph_height > 0:
        scale_x = (1.0 - margin) / graph_width
        scale_y = (1.0 - margin) / graph_height
        scale_factor = min(scale_x, scale_y)

    for node in pos:
        x, y = pos[node]
        pos[node] = (x * scale_factor, y * scale_factor)

    min_x_scaled = min(x for x, y in pos.values())
    max_x_scaled = max(x for x, y in pos.values())
    min_y_scaled = min(y for x, y in pos.values())
    max_y_scaled = max(y for x, y in pos.values())

    center_x_graph = (min_x_scaled + max_x_scaled) / 2
    center_y_graph = (min_y_scaled + max_y_scaled) / 2
    center_x_image = 0.5
    center_y_image = 0.5

    offset_x = center_x_image - center_x_graph
    offset_y = center_y_image - center_y_graph

    image_width = background_image.width
    image_height = background_image.height

    base_node_radius = 20
    node_radius = int(base_node_radius * scale_factor)
    edge_color = (128, 128, 128)
    edge_base_width = 4.5
    edge_width = max(1, int(edge_base_width * scale_factor))
    best_path_edge_color = (255, 0, 0)
    best_path_edge_width = max(1, int(edge_base_width * 1.5 * scale_factor))
    start_node_color = (0, 255, 0)
    default_node_color = (173, 216, 230)
    font_color = (0, 0, 0)
    base_font_size = 104
    try:
        font = ImageFont.truetype("arial.ttf", int(base_font_size * scale_factor))
    except IOError:
        font = ImageFont.load_default()

    for edge in G.edges():
        u, v = edge
        x1, y1 = pos[u]
        x2, y2 = pos[v]
        x1_centered = x1 + offset_x
        y1_centered = y1 + offset_y
        x2_centered = x2 + offset_x
        y2_centered = y2 + offset_y
        start_point_px = (int(x1_centered * image_width), int(y1_centered * image_height))
        end_point_px = (int(x2_centered * image_width), int(y2_centered * image_height))

        current_edge_color = edge_color
        current_edge_width = edge_width

        if best_path:
            path_edges = list(zip(best_path,best_path[1:])) + [(best_path[-1],best_path[0])]
            if (u,v) in path_edges or (v,u) in path_edges:
                current_edge_color = best_path_edge_color
                current_edge_width = best_path_edge_width

        draw.line([start_point_px, end_point_px], fill=current_edge_color, width=current_edge_width)

    for node in G.nodes():
        x, y = pos[node]
        x_centered = x + offset_x
        y_centered = y + offset_y
        node_center = (int(x_centered * image_width), int(y_centered * image_height))
        node_bbox = (node_center[0] - node_radius, node_center[1] - node_radius,
                     node_center[0] + node_radius, node_center[1] + node_radius)
        node_color = default_node_color
        if node == start_point:
            node_color = start_node_color
        draw.ellipse(node_bbox, fill=node_color)
        draw.text(node_center, str(node), fill=font_color, font=font, anchor="mm")

    background_image.save(output_image_path)
    print(f"Graphe sauvegardé dans : {output_image_path}")
    return output_image_path

def visualize_road_network(G, start_point, sources, seed, best_path=None):
    image_path = visualize_road_network_pillow(G, start_point, sources, best_path=best_path)
    st.image(image_path)

def choose_random_start_and_sources(G):
    start_point = random.choice(list(G.nodes))
    sources = []
    return start_point, sources

time_step = 0.5

# --- Paramètres de congestion aux heures de pointe ---
peak_hour_morning_start = 8 * 3600  # 8h en secondes
peak_hour_morning_end = 9 * 3600    # 9h en secondes
peak_hour_lunch_start = 12 * 3600   # 12h en secondes
peak_hour_lunch_end = 13 * 3600     # 13h en secondes
peak_hour_evening_start = 17 * 3600  # 17h en secondes
peak_hour_evening_end = 19 * 3600    # 19h en secondes


def gaussian(t, mu, sigma, A):
    return A * math.exp(-((t - mu)**2) / (2 * sigma**2))

def amplitude_spatiale_gaussienne(x, road_length, amplitude_max, sigma_spatial_fraction=0.3):
    sigma_spatial = road_length * sigma_spatial_fraction
    return amplitude_max * math.exp(-((x - road_length/2)**2) / (2 * sigma_spatial**2))

sigma_hours_peak1 = 0.75
sigma_hours_peak2 = 1.0
sigma_hours_peak3 = 0.75
peak1_hour = 8
peak2_hour = 17
peak3_hour = 13

sigma_seconds_peak1 = sigma_hours_peak1 * 3600
sigma_seconds_peak2 = sigma_hours_peak2 * 3600
sigma_seconds_peak3 = sigma_hours_peak3 * 3600
peak1_seconds = peak1_hour * 3600
peak2_seconds = peak2_hour * 3600
peak3_seconds = peak3_hour * 3600

def rho_gaussiennes_spatiale(x, t, road_length, rho_max, rho_base_fraction=0.1, amplitude_max_fraction=0.7,
                             sigma_seconds_peak1=sigma_seconds_peak1, sigma_seconds_peak2=sigma_seconds_peak2, sigma_seconds_peak3=sigma_seconds_peak3,
                             peak1_seconds=peak1_seconds, peak2_seconds=peak2_seconds, peak3_seconds=peak3_seconds,
                             sigma_spatial_fraction=0.3):
    rho_base = rho_max * rho_base_fraction
    amplitude_max = rho_max * amplitude_max_fraction
    sigma1 = sigma_seconds_peak1
    sigma2 = sigma_seconds_peak2
    sigma3 = sigma_seconds_peak3

    amplitude_x = amplitude_spatiale_gaussienne(x, road_length, amplitude_max, sigma_spatial_fraction)

    peak1 = gaussian(t, peak1_seconds, sigma1, amplitude_x)
    peak2 = gaussian(t, peak2_seconds, sigma2, amplitude_x)
    peak3 = gaussian(t, peak3_seconds, sigma3, amplitude_x)

    rho_t = rho_base + peak1 + peak2 + peak3
    return max(0, min(rho_t, rho_max))

def velocity_spatial(x, t, road_length, v_max, rho_max, road_length_param):
    rho_value = rho_gaussiennes_spatiale(x, t, road_length_param, rho_max,
                                        sigma_seconds_peak1=sigma_seconds_peak1, sigma_seconds_peak2=sigma_seconds_peak2, sigma_seconds_peak3=sigma_seconds_peak3,
                                        peak1_seconds=peak1_seconds, peak2_seconds=peak2_seconds, peak3_seconds=peak3_seconds)
    return v_max * (1 - rho_value / rho_max)

def simulate_vehicle_spatial(road_length, v_max, route_rho_max, road_length_param, time_step, journey_start_time=None):
    position = 0
    time_elapsed_in_simulation = 0
    segment_travel_time = 0

    while position < road_length:
        current_time_for_density = journey_start_time + time_elapsed_in_simulation if journey_start_time is not None else time_elapsed_in_simulation
        v = velocity_spatial(position, current_time_for_density, road_length, v_max, route_rho_max, road_length_param)
        if v <= 0:
            return float('inf')
        position += v * time_step
        time_elapsed_in_simulation += time_step
        segment_travel_time += time_step

    return segment_travel_time

def simulate_journey(start_node, end_node, start_time, matrix, v_max, G, time_step):
    current_node = start_node
    current_time = start_time
    total_time = 0.0
    path_nodes = [start_node]

    while current_node != end_node:
        next_node = end_node
        if next_node not in G.neighbors(current_node):
            return float('inf')

        road_length = G[current_node][next_node]['length']
        route_rho_max = G[current_node][next_node]['rho_max']
        road_length_param = road_length

        segment_time = simulate_vehicle_spatial(road_length=road_length, v_max=v_max, route_rho_max=route_rho_max, road_length_param=road_length_param, time_step=time_step, journey_start_time=current_time)

        if segment_time == float('inf'):
            return float('inf')

        current_time += segment_time
        total_time += segment_time
        current_node = next_node
        path_nodes.append(next_node)

        if current_node == end_node:
            break

    return total_time

@st.cache_data
def generate_graph_data(taille_matrice, borne_min, borne_max, rho_max_global, congestion_factor_morning_peak, congestion_factor_lunch_peak, congestion_factor_evening_peak, mu_T, sigma_T):
    adj_matrix = generer_matrice_uniforme((taille_matrice, taille_matrice), borne_min, borne_max)
    G = nx.from_numpy_array(adj_matrix)
    start_point, sources = choose_random_start_and_sources(G)
    road_network = nx.complete_graph(taille_matrice)
    n = len(road_network.nodes())
    matrix = np.full((n, n), float('inf'))

    for u, v in road_network.edges():
        weight = int(np.random.uniform(borne_min, borne_max))
        matrix[u, v] = weight
        matrix[v, u] = weight
        base_rho_max = np.random.uniform(rho_max_global * 0.3, rho_max_global)

        # --- Facteurs de congestion appliqués to rho_max directly ---
        peak_hour_start_morning = 7
        peak_hour_end_morning = 9
        peak_hour_start_evening = 16
        peak_hour_end_evening = 18

        is_peak_hour_morning = (peak_hour_start_morning <= peak1_hour < peak_hour_end_morning)
        is_peak_hour_lunch = (peak_hour_lunch_start <= peak3_hour < peak_hour_lunch_end)
        is_peak_hour_evening = (peak_hour_start_evening <= peak2_hour < peak_hour_end_evening)

        congestion_factor = 1.0
        if is_peak_hour_morning:
            congestion_factor = max(congestion_factor, congestion_factor_morning_peak)
        if is_peak_hour_lunch:
            congestion_factor = max(congestion_factor, congestion_factor_lunch_peak)
        if is_peak_hour_evening:
            congestion_factor = max(congestion_factor, congestion_factor_evening_peak)

        road_importance_factor = 1.0
        max_possible_distance = borne_max * 2
        normalized_distance = matrix[u, v] / max_possible_distance
        road_importance_congestion_factor = max(0.5, 2 - normalized_distance * 3)

        combined_congestion_factor = congestion_factor * road_importance_congestion_factor
        route_rho_max = base_rho_max * combined_congestion_factor

        route_period = int(max(1, np.random.normal(mu_T, sigma_T)))

        # --- Ajout des attributs à l'arête du graphe ---
        G.add_edge(u, v, length=weight, rho_max=route_rho_max, period=route_period)
        G.add_edge(v, u, length=weight, rho_max=route_rho_max, period=route_period)

    np.fill_diagonal(matrix, 0)

    time_matrix = None
    rho_max_matrix = None

    seed_value = random.randint(0, 1000)
    return adj_matrix, G, start_point, sources, matrix, None, time_matrix, rho_max_matrix, seed_value

# --- Interface utilisateur Streamlit ---
st.title("Optimisation du Transport dans les Réseaux Citadins (Quand la Fourmi est au Volant)")
st.markdown("""Cette application permet de simuler et optimiser les flux de transport par l'Algorithme des Colonies des Fourmis 🐜🐜🐜.""")

ville_choisie = st.selectbox("Choisir une ville (préréglages de trafic):",
                            list(city_traffic_params.keys()), index=1)

if ville_choisie == "Personnalisé":
    st.sidebar.header("Paramètres de Trafic Personnalisés")
    rho_max_global_user = st.sidebar.slider("Densité Maximale Globale", 100, 500, city_traffic_params["Personnalisé"]["rho_max_global"])
    congestion_factor_morning_peak_user = st.sidebar.slider("Facteur Congestion Matin", 1.0, 3.0, city_traffic_params["Personnalisé"]["congestion_factor_morning_peak"], step=0.1)
    congestion_factor_lunch_peak_user = st.sidebar.slider("Facteur Congestion Déjeuner", 1.0, 2.0, city_traffic_params["Personnalisé"]["congestion_factor_lunch_peak"], step=0.1)
    congestion_factor_evening_peak_user = st.sidebar.slider("Facteur Congestion Soir", 1.0, 3.0, city_traffic_params["Personnalisé"]["congestion_factor_evening_peak"], step=0.1)
    v_max_user = st.sidebar.slider("Vitesse Maximale (km/h)", 10, 30, int(city_traffic_params["Personnalisé"]["v_max"]), step=1)
    mu_T_user = st.sidebar.slider("Mu (Temps Pic Trafic, secondes)", 0, 86400, city_traffic_params["Personnalisé"]["mu_T"], step=3600)
    sigma_T_user = st.sidebar.slider("Sigma (Étalement Pic Trafic, secondes)", 1800, 14400, city_traffic_params["Personnalisé"]["sigma_T"], step=3600)

    rho_max_global = rho_max_global_user
    congestion_factor_morning_peak = congestion_factor_morning_peak_user
    congestion_factor_lunch_peak = congestion_factor_lunch_peak_user
    congestion_factor_evening_peak = congestion_factor_evening_peak_user
    v_max = v_max_user
    mu_T = mu_T_user
    sigma_T = sigma_T_user

else:
    params_ville = city_traffic_params[ville_choisie]
    rho_max_global = params_ville["rho_max_global"]
    congestion_factor_morning_peak = params_ville["congestion_factor_morning_peak"]
    congestion_factor_lunch_peak = params_ville["congestion_factor_lunch_peak"]
    congestion_factor_evening_peak = params_ville["congestion_factor_evening_peak"]
    v_max = params_ville["v_max"]
    mu_T = params_ville["mu_T"]
    sigma_T = params_ville["sigma_T"]


taille_matrice = st.slider('Nombre de Stations', 5, 50, 20)
borne_min = st.slider('Distance Minimale du Réseau', 1000, 10000, 2000)
borne_max = st.slider('Distance Maximale du Réseau', 1000, 10000, 8000)

graph_data = generate_graph_data(taille_matrice, borne_min, borne_max, rho_max_global, congestion_factor_morning_peak, congestion_factor_lunch_peak, congestion_factor_evening_peak, mu_T, sigma_T)
adj_matrix, G, start_point, sources, matrix, period_matrix, time_matrix, rho_max_matrix, seed_value = graph_data

st.write(f"Ville choisie : **{ville_choisie}**")
st.write(f"Point de départ : {start_point}")
visualize_road_network(G, start_point, sources, seed_value)


if 'show_distance_matrix' not in st.session_state:
    st.session_state.show_distance_matrix = False
if st.button("Visualiser/Désafficher la Matrice des Distances"):
    st.session_state.show_distance_matrix = not st.session_state.show_distance_matrix
if st.session_state.show_distance_matrix:
    st.write("Matrice des distances (en mètres) :")
    st.dataframe(matrix)

if 'show_time_matrix' not in st.session_state:
    st.session_state.show_time_matrix = False

st.markdown("### L'expression de la vitesse :")
st.latex(r'''v(x, t) = v_{\text{max}} \cdot \left(1 - \frac{\rho(x, t)}{\rho_{\text{max}}}\right)''')
st.markdown("Paramètres :")
st.latex(r'''v_{\text{max}} : \text{Vitesse maximale possible dans la ville}''')
st.latex(r'''\rho(x, t) : \text{Densité de trafic calculée (selon la ville et l'heure)}''')
st.latex(r'''\rho_{\text{max}} : \text{Densité maximale de la route (caractéristique de la ville)}''')

st.markdown("### Paramètres de Pondération")
alpha = st.selectbox("Choisir le poids de la distance (PdD) :", [round(i * 0.1, 1) for i in range(11)], index=5)
beta = st.selectbox("Choisir le poids du temps (PdT) :", [round(i * 0.1, 1) for i in range(11)], index=5)
st.success(f"Les valeurs sélectionnées sont : PdD = {alpha} et PdT = {beta}")

st.markdown("### Paramètres Temporels")
departure_time = st.time_input("Heure de départ du trajet :", datetime.time(8, 00)) # Heure par défaut : 8h00
user_start_time_seconds = departure_time.hour * 3600 + departure_time.minute * 60
st.write(f"Heure de départ choisie : {departure_time.strftime('%H:%M')}") # Affichage formaté HH:MM


n = len(matrix)
cost_matrix = np.full((n, n), float('inf'))
time_matrix_user_time = np.full((n, n), float('inf'))

with st.spinner('Calcul de la matrice des temps de parcours...'):
    for u in range(n):
        for v in range(n):
            if u != v:
                travel_time = simulate_journey(u, v, user_start_time_seconds, matrix, v_max, G, time_step)
                time_matrix_user_time[u, v] = travel_time
                time_matrix_user_time[v, u] = time_matrix_user_time[u, v] # Assuming symmetric times, remove if not the case
                cost_matrix[u, v] = alpha * matrix[u, v] + beta * time_matrix_user_time[u, v]
                cost_matrix[v, u] = cost_matrix[u, v] # Assuming symmetric costs
    np.fill_diagonal(cost_matrix, 0)
    np.fill_diagonal(time_matrix_user_time, 0)
st.success("Matrice des temps de parcours calculée!")


if st.button("Visualiser/Désafficher la Matrice des Temps de Parcours (pour l'heure choisie)"):
    st.session_state.show_time_matrix = not st.session_state.show_time_matrix
if st.session_state.show_time_matrix:
    st.write("Matrice des temps de parcours (pour l'heure choisie) :")
    st.dataframe(time_matrix_user_time)



def initialize_pheromones(cost_matrix, initial_value=1000.0):
    n = len(cost_matrix)
    pheromones = np.full((n, n), initial_value)
    return pheromones

def calculate_transition_probabilities(pheromones, costs, alpha, beta, visited):
    probabilities = []
    valid_indices = []

    for i in range(len(costs)):
        if i in visited:
            probabilities.append(0)
        else:
            if costs[i] > 0 and costs[i] != float('inf') and pheromones[i] > 0:
                prob = (pheromones[i] ** alpha) * ((1 / costs[i]) ** beta)
                probabilities.append(prob)
                valid_indices.append(i)
            else:
                probabilities.append(0)

    probabilities = np.array(probabilities)
    prob_sum = probabilities.sum()

    if prob_sum == 0:
        non_visited_nodes = [i for i in range(len(costs)) if i not in visited]
        if non_visited_nodes:
            default_probs = np.zeros(len(costs))
            for index in non_visited_nodes:
                default_probs[index] = 1.0 / len(non_visited_nodes)
            return default_probs
        else:
            return np.ones(len(probabilities)) / len(probabilities)

    probabilities /= prob_sum
    return probabilities

def preprocess_matrix(matrix):
    n = len(matrix)
    processed_matrix = np.copy(matrix)
    for i in range(n):
        for j in range(n):
            if i != j and processed_matrix[i, j] == 0:
                processed_matrix[i, j] = float('inf')
    return processed_matrix

def ant_colony_optimization(cost_matrix, time_matrix, distance_matrix, num_ants, num_iterations, alpha_aco, beta_aco, evaporation_rate, q, start_point_aco):
    n = len(cost_matrix)
    pheromones = initialize_pheromones(cost_matrix)
    best_path = None
    best_distance = float('inf')
    best_time = float('inf')

    for iteration in range(num_iterations):
        all_paths = []
        all_distances = []
        all_times = []

        for ant in range(num_ants):
            current_node = start_point_aco
            visited = [current_node]
            path_distance = 0
            path_time = 0

            while len(visited) < n:
                probabilities = calculate_transition_probabilities(
                    pheromones[current_node], cost_matrix[current_node], alpha_aco, beta_aco, visited
                )
                next_node = np.random.choice(range(n), p=probabilities)
                path_distance += distance_matrix[current_node, next_node]
                path_time += time_matrix[current_node, next_node]
                visited.append(next_node)
                current_node = next_node

            path_distance += distance_matrix[current_node, visited[0]]
            path_time += time_matrix[current_node, visited[0]]
            visited.append(visited[0])

            all_paths.append(visited)
            all_distances.append(path_distance)
            all_times.append(path_time)

            if path_distance < best_distance:
                best_path = visited
                best_distance = path_distance
                best_time = path_time

        pheromones *= (1 - evaporation_rate)
        for path, distance in zip(all_paths, all_distances):
            path_cost = 0
            for i in range(len(path) - 1):
                path_cost += cost_matrix[path[i], path[i+1]]
            for i in range(len(path) - 1):
                pheromones[path[i], path[i + 1]] += q / path_cost

    return best_path, best_distance, best_time

st.markdown("### Paramètres Ant Colony Optimization (ACO)")
col1, col2, col3, col4, col5, col6 = st.columns(6)
num_ants = col1.number_input("Fourmis:", min_value=1, value=10)
num_iterations = col2.number_input("Iterations:", min_value=1, value=100)
alpha_aco = col3.number_input("Alpha (phéromones):", min_value=0.0, value=1.0)
beta_aco = col4.number_input("Beta (coût):", min_value=0.0, value=2.0)
evaporation_rate = col5.number_input("Evaporation:", min_value=0.0, value=0.5)
q = col6.number_input("Q:", min_value=0.0, value=100.0)
start_point_aco = start_point
if st.button("Lancer ACO", key="run_aco_button"):
    processed_cost_matrix = preprocess_matrix(cost_matrix)
    with st.spinner('Optimisation du chemin...'):
        best_path, best_distance, best_time = ant_colony_optimization(
            processed_cost_matrix, time_matrix_user_time, matrix, num_ants, num_iterations, alpha_aco, beta_aco, evaporation_rate, q, start_point_aco
        )
    st.success('Optimisation terminée!')
    st.write("Meilleur Chemin Trouvé (nœuds visités dans l'ordre):", best_path)
    best_distance_km = round(best_distance / 1000, 2)
    best_time_hours = round(best_time / 3600, 2)
    st.write(f"Distance Totale du Chemin: {best_distance_km} km")
    st.write(f"Temps Total de Parcours: {best_time_hours} heures")

    visualize_road_network(G, start_point, sources, seed_value, best_path)
