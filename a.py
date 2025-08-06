# streamlit_app.py

import streamlit as st
import os
import requests
from dotenv import load_dotenv
from io import StringIO

from langchain_tavily import TavilySearch
from langchain_core.tools import tool
from langchain.agents import create_react_agent, AgentExecutor
from langchain import hub
from langchain_google_genai import ChatGoogleGenerativeAI
import warnings
from langsmith.utils import LangSmithMissingAPIKeyWarning
warnings.filterwarnings("ignore", category=LangSmithMissingAPIKeyWarning)


# ---------------------------
# Load environment variables
# ---------------------------
load_dotenv()
tavily_api_key = os.getenv("TAVILY_API_KEY")
open_weather_api_key = os.getenv("OPEN_WEATHER_API_KEY")
google_api_key = os.getenv("GOOGLE_API_KEY")

# ---------------------------
# Initialize LLM
# ---------------------------
llm = ChatGoogleGenerativeAI(
    model="gemini-1.5-flash",
    temperature=0.2,
    max_output_tokens=1500,
    api_key=google_api_key
)

# ---------------------------
# Define tools
# ---------------------------
@tool
def tavily_search(query: str) -> str:
    """Search the web using Tavily and return summarized content from top results."""
    tavily_tool = TavilySearch(
        max_results=5,
        topic="general",
        api_key=tavily_api_key,
    )
    response = tavily_tool.invoke(query)
    combined_content = ""
    for i, result in enumerate(response['results'], 1):
        if 'content' in result and result['content']:
            combined_content += f"{result['content']}\n"
    return combined_content.strip()

@tool
def get_weather(location: str) -> str:
    """Get the current weather for a given location."""
    if not open_weather_api_key:
        raise ValueError("OpenWeather API key is not set.")
    url = f"http://api.openweathermap.org/data/2.5/weather?q={location}&appid={open_weather_api_key}&units=metric"
    response = requests.get(url)
    data = response.json()
    temp = data['main']['temp']
    description = data['weather'][0]['description']
    return f"The current temperature in {location} is {temp}°C with {description}."

@tool
def get_hotel_recommendations(location: str) -> str:
    """Get hotel recommendations for a given location."""
    search_tool = TavilySearch(
        max_results=5,
        topic="general", 
        api_key=tavily_api_key,
    )
    response = search_tool.invoke(f"Best hotels in {location}")
    return response['results'][0]['content'] if response['results'] else "No hotel recommendations found."

# ---------------------------
# Agent setup
# ---------------------------
tools = [tavily_search, get_weather, get_hotel_recommendations]
prompt = hub.pull("hwchase17/react")
agent = create_react_agent(llm=llm, tools=tools, prompt=prompt)
agent_executor = AgentExecutor(
    agent=agent,
    tools=tools,
    verbose=False,
    max_iterations=15,
    max_execution_time=120
)

## ---------------------------
# Streamlit UI
# ---------------------------
st.set_page_config(page_title="AI Travel Planner", page_icon="🌍", layout="wide")

# Title and description
st.title("🌍 AI Travel Planner")
st.subheader("Plan your perfect trip with AI-powered suggestions.")
st.markdown("Fill in the travel details in the sidebar and let the AI generate a personalized, day-by-day travel itinerary.")

# Sidebar for input
import streamlit as st

# Initialize default values only if not already set
if "city_input" not in st.session_state:
    st.session_state.city_input = ""
if "duration_input" not in st.session_state:
    st.session_state.duration_input = 1
if "interests_input" not in st.session_state:
    st.session_state.interests_input = ""
if "time_input" not in st.session_state:
    st.session_state.time_input = ""


# Clear inputs button logic
def clear_inputs():
    st.session_state.city_input = ""
    st.session_state.duration_input = 1
    st.session_state.interests_input = ""
    st.session_state.time_input = ""

with st.sidebar:
    st.header("📝 Trip Details")

    city = st.text_input("City", placeholder="e.g., Kolhapur", key="city_input")
    duration = st.number_input("Duration (in days)", min_value=1, max_value=30, key="duration_input")
    interests = st.text_input("Your Interests", placeholder="e.g., Food, history, shopping", key="interests_input")
    time_of_year = st.text_input("Time of Year", placeholder="e.g., August", key="time_input")

    submitted = st.button("Generate Travel Plan")
    clear = st.button("Clear Inputs", on_click=clear_inputs)

# Use these variables as usual for your logic:
city = st.session_state.city_input
duration = st.session_state.duration_input
interests = st.session_state.interests_input
time_of_year = st.session_state.time_input


# Generate trip plan
if submitted:
    if not city or not interests or not time_of_year:
        st.warning("Please fill in all required fields.")
    else:
        user_query = (
            f"Plan a {duration}-day trip to {city} in {time_of_year}, focusing on {interests}. "
            "Break the itinerary into daily activities in paragraph format. Include travel suggestions, sightseeing, food, and cultural experiences. "
            "Also include climate information and typical temperature ranges for that time of year. "
            "Present each day separately and clearly labeled (Day 1, Day 2, etc)."
        )

        input_text = (
            f"You are an expert travel planner. Your task is to create a detailed travel plan for the user based on their preferences and the provided city. "
            f"Include top restaurants {city} has to offer and the best places to visit. Include weather, climate, and typical temperature information for {city} during {time_of_year}. "
            f"\n\nUser query: {user_query}\n\nGenerate the response using the agent executor."
        )

        with st.spinner("🧠 Planning your personalized trip..."):
            try:
                response = agent_executor.invoke({"input": user_query})
                plan_text = response["output"]

                # Display result
                st.success("✅ Here's your travel plan:")

                # Split into days for display
                days = plan_text.split("Day ")
                for i, day_plan in enumerate(days[1:], start=1):
                    with st.expander(f"📅 Day {i}", expanded=True):
                        st.write(f"**Day {i}**: {day_plan.strip()}")

                from io import BytesIO

                plan_bytes = BytesIO()
                plan_bytes.write(f"AI Travel Plan for {city}\n\n{plan_text}".encode("utf-8"))
                plan_bytes.seek(0)

                st.download_button(
                label="📥 Download Travel Plan",
                data=plan_bytes,
                file_name=f"{city}_travel_plan.txt",
                mime="text/plain"
                )


            except Exception as e:
                st.error(f"❌ Something went wrong: {e}")
