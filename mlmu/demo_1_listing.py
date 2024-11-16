import openai
from pydantic import BaseModel, Field

listing_text = """\
For sale:
I have a loaded 14-inch MacBook Pro M1 Max, 64 GB RAM, 1 TB SSD, Space Gray. This powerful machine 
is perfect for professionals, students, or anyone needing a powerful and fast laptop for gaming 
or running AI applications. In fact, this M1 Max has better GPU performance than even the M2 Pro.
A new M3/M4 Max with 64GB would run you over $5000 including tax

Specs:
* Model: MacBook Pro, Space Gray (Z15G001XD)
* Memory/RAM: 64 GB
* Storage: 1 TB Storage
* Processor: M1 Max with 10-core CPU and 24-core GPU
* Display: 14.2-inch (diagonal) Liquid Retina XDR; 3024-by-1964 native resolution
* Three Thunderbolt 4 (USB-C) ports for charging, DisplayPort, USB 4

Features:
* Fast and responsive performance
* Ample storage for all your files and applications
* Crisp and clear Retina display
* Up to 17 hours battery
* Active AppleCare+ coverage until Feb, 2025
* Additional specs: https://support.apple.com/en-us/111902 (the one listed here has M1 Max, RAM 
  and Storage upgrades)

Condition:
* Like new, hardly used with no scratches, dents or dead pixels

Included:
* Apple original laptop box and all original MagSafe 3 charging accessories
"""


class LaptopListing(BaseModel):
    manufacturer: str = Field(description="Laptop manufacturer, e.g. Apple, Acer, Lenovo, etc.")
    storage_size: int = Field(description="Storage size in GB")
    cpu_manufacturer: str = Field(description="CPU manufacturer, e.g. Intel, AMD, Samsung")
    ram_size: int = Field(description="RAM size in GB")


client = openai.OpenAI()

prompt = f"""\
You are an expert on analyzing Craigslist laptop listings and turning them into JSON. 
Analyze the following listing and return a JSON object containing information about the
laptop described in the listing:

<listing>
{listing_text}
</listing>
"""

completion = client.beta.chat.completions.parse(
    model="gpt-4o-mini",
    messages=[{"role": "user", "content": prompt}],
    response_format=LaptopListing,
    temperature=0
)

extracted_listing = completion.choices[0].message.parsed

assert extracted_listing is not None
assert extracted_listing.manufacturer == "Apple"
assert extracted_listing.storage_size == 1000
assert extracted_listing.cpu_manufacturer == "Apple"
assert extracted_listing.ram_size == 64

print(extracted_listing)
