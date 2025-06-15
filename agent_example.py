from crewai import Agent, Task, Crew, Process, LLM

# Initialize Ollama with local setup
ollama = LLM(
    base_url="http://localhost:11434/api/generate",
    model="ollama/gemma3:1b"
)

# Define our specialized blog writing agents
researcher = Agent(
    role='Blog Research Specialist',
    goal='Research and gather comprehensive information about the given topic',
    backstory="""You are an expert researcher with a keen eye for detail and the ability to prepare 
    the most relevant and interesting information about any topic. You excel at organizing information 
    in a way that will be useful for writing engaging blog posts.""",
    verbose=True,
    llm=ollama
)

writer = Agent(
    role='Professional Blog Writer',
    goal='Create engaging and well-structured blog posts',
    backstory="""You are a skilled blog writer known for creating compelling content that engages 
    readers while maintaining high standards of clarity and informativeness. You have a talent for 
    turning complex topics into accessible and interesting blog posts.""",
    verbose=True,
    llm=ollama
)

editor = Agent(
    role='Content Editor',
    goal='Polish and refine blog content for maximum impact',
    backstory="""You are a meticulous editor with years of experience in digital content. 
    You excel at improving content structure, readability, and ensuring the final piece meets 
    high-quality standards while maintaining SEO best practices.""",
    verbose=True,
    llm=ollama
)

def create_blog_post(topic):
    """
    Create a blog post about the given topic using our crew of agents.
    """
    # Define tasks for our blog creation process
    research_task = Task(
        description=f"Research the topic: {topic}. Prepare key information, interesting facts, and relevant data. Limit to 200 words",
        agent=researcher,
        expected_output="Detailed research notes and key points for the blog post"
    )

    writing_task = Task(
        description=f"Write a comprehensive blog post about {topic} using the research provided. "
                   f"Create an engaging introduction, well-structured body with clear sections, "
                   f"and a compelling conclusion.",
        agent=writer,
        expected_output="Complete first draft of the blog post"
    )

    editing_task = Task(
        description="Review and polish the blog post. Check for clarity, flow, and engagement. "
                   "Ensure proper structure, formatting, and optimal readability.",
        agent=editor,
        expected_output="Final, polished version of the blog post"
    )

    # Create and run our crew
    crew = Crew(
        agents=[researcher, writer, editor],
        tasks=[research_task, writing_task, editing_task],
        verbose=True,
        process=Process.sequential
    )

    # Execute the blog creation process
    result = crew.kickoff()
    return result

if __name__ == "__main__":
    # Example usage
    topic = "The Future of Artificial Intelligence in Healthcare"
    blog_post = create_blog_post(topic)
    
    # Create a filename based on the topic
    filename = f"blog_post_{topic.lower().replace(' ', '_')}.txt"
    
    # Save the blog post to a file - convert CrewOutput to string
    with open(filename, 'w') as f:
        f.write(str(blog_post))
    
    print("\nFinal Blog Post:")
    print(blog_post)
    print(f"\nBlog post has been saved to: {filename}")