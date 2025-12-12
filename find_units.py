"""
Find Unit Names and Structure in the Textbook
"""

import os
import fitz
import base64
import requests
from dotenv import load_dotenv

load_dotenv()

def find_units_in_textbook():
    """Search for unit names in the textbook"""
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        print("❌ Please set OPENAI_API_KEY in your .env file")
        return

    pdf_path = "Secondary - 2018 - Class - 9&10 - English for today-9-10  PDF Web .pdf"
    document = fitz.open(pdf_path)
    total_pages = len(document)
    document.close()

    print(f"📖 Searching for units in textbook (Total pages: {total_pages})")
    print("Checking pages typically containing table of contents...")

    # Check pages that might contain unit information
    pages_to_check = list(range(1, min(30, total_pages)))  # First 30 pages

    for page_num in pages_to_check:
        print(f"\nChecking page {page_num}...")

        try:
            document = fitz.open(pdf_path)
            page = document[page_num - 1]

            # Convert to image
            mat = fitz.Matrix(2.0, 2.0)
            pix = page.get_pixmap(matrix=mat)
            img_data = pix.tobytes("png")
            base64_image = base64.b64encode(img_data).decode('utf-8')
            document.close()

            headers = {
                "Content-Type": "application/json",
                "Authorization": f"Bearer {api_key}"
            }

            payload = {
                "model": "gpt-4o",
                "messages": [
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "text",
                                "text": """Extract any unit information from this page. Look for:
                                - "Unit" followed by a number (e.g., "Unit 1", "Unit 2")
                                - Unit titles or names
                                - Lesson names within units
                                - Table of contents
                                - Index or contents page

                                If you find any units, list them clearly with their names and numbers."""
                            },
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:image/png;base64,{base64_image}"
                                }
                            }
                        ]
                    }
                ],
                "max_tokens": 1000
            }

            response = requests.post(
                "https://api.openai.com/v1/chat/completions",
                headers=headers,
                json=payload,
                timeout=30
            )

            if response.status_code == 200:
                result = response.json()['choices'][0]['message']['content']

                # Check if this contains unit information
                if any(keyword.lower() in result.lower() for keyword in ['unit', 'lesson', 'content']):
                    print(f"  ✅ Found relevant content on page {page_num}:")
                    print(f"  {result[:300]}...")

                    # Save if this looks like a contents page
                    if 'unit' in result.lower() and any(num in result for num in ['1', '2', '3']):
                        with open(f"page_{page_num}_contents.txt", "w") as f:
                            f.write(result)
                        print(f"  💾 Saved to page_{page_num}_contents.txt")
                        break  # Stop after finding the contents page
                else:
                    print(f"  - No unit information found")
            else:
                print(f"  ❌ Error: {response.status_code}")

        except Exception as e:
            print(f"  ❌ Error processing page: {e}")

    print("\n✅ Search completed!")
    print("\nTo find units more efficiently, try pages 5-15 which often contain the table of contents.")


if __name__ == "__main__":
    find_units_in_textbook()