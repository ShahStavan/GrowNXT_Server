import os
import asyncio
from pathlib import Path
from typing import Optional
import time
import base64
import io
from PIL import Image
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.common.exceptions import TimeoutException, NoSuchElementException
from config import REQUIRED_ENV_VARS
from langchain_google_genai import ChatGoogleGenerativeAI
from pydantic import SecretStr
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()


class CaptchaSolver:
    """A class to handle Google reCAPTCHA challenges during web scraping."""
    
    def __init__(self, driver: webdriver.Chrome, api_key: Optional[str] = None):
        self.driver = driver
        self.api_key = api_key or os.getenv('GEMINI_API_KEY')
        if not self.api_key:
            raise ValueError("GEMINI_API_KEY environment variable is not set")
        
        self.llm = ChatGoogleGenerativeAI(
            model='gemini-1.5-flash',  # Using a model suitable for visual tasks
            api_key=SecretStr(self.api_key)
        )
    
    def detect_captcha(self) -> bool:
        """Detect if a CAPTCHA is present on the page."""
        try:
            # Check for reCAPTCHA iframe
            captcha_frames = self.driver.find_elements(
                By.CSS_SELECTOR, 
                "iframe[src*='recaptcha'], iframe[title*='recaptcha'], iframe[title*='reCAPTCHA']"
            )
            
            if captcha_frames:
                print("CAPTCHA detected on page")
                return True
                
            # Also check for text indicators
            page_source = self.driver.page_source.lower()
            captcha_indicators = [
                "captcha", "i'm not a robot", "human verification",
                "verify you are human", "security check"
            ]
            
            if any(indicator in page_source for indicator in captcha_indicators):
                print("CAPTCHA or verification check detected on page")
                return True
                
            return False
            
        except Exception as e:
            print(f"Error detecting CAPTCHA: {e}")
            return False
    
    def solve_image_captcha(self):
        """Solve image-based CAPTCHA challenges using Gemini Vision API."""
        try:
            # Check if there are image challenges
            image_frames = self.driver.find_elements(
                By.CSS_SELECTOR, 
                "iframe[src*='recaptcha/api2/bframe']"
            )
            
            if not image_frames:
                return False
                
            # Switch to the image challenge frame
            self.driver.switch_to.frame(image_frames[0])
            
            # Wait for the image challenge to load
            WebDriverWait(self.driver, 10).until(
                EC.presence_of_element_located((By.CSS_SELECTOR, ".rc-imageselect-desc"))
            )
            
            # Get the instruction text
            instruction_element = self.driver.find_element(By.CSS_SELECTOR, ".rc-imageselect-desc")
            instruction = instruction_element.text
            print(f"CAPTCHA challenge: {instruction}")
            
            # Take a screenshot of the CAPTCHA grid
            captcha_grid = self.driver.find_element(By.CSS_SELECTOR, ".rc-imageselect-target")
            screenshot = captcha_grid.screenshot_as_base64
            
            # Convert the screenshot to a PIL Image
            image_data = base64.b64decode(screenshot)
            image = Image.open(io.BytesIO(image_data))
            
            # Use Gemini to analyze the image and get instructions on which tiles to click
            prompt = f"""
            I'm facing a reCAPTCHA challenge that says: "{instruction}".
            Look at this image grid and tell me which numbered tiles I should click on.
            Number the tiles from 1-9, starting from the top-left and going row by row.
            Just give me the numbers of tiles to click, separated by commas.
            """
            
            # Call Gemini Vision API to analyze the image
            response = self.llm.invoke(
                [{"type": "text", "text": prompt}, 
                 {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{screenshot}"}}]
            )
            
            tile_numbers = self._parse_gemini_response(response.content)
            print(f"AI suggests clicking tiles: {tile_numbers}")
            
            # Click on the suggested tiles
            tiles = self.driver.find_elements(By.CSS_SELECTOR, ".rc-imageselect-tile")
            for tile_num in tile_numbers:
                if 1 <= tile_num <= len(tiles):
                    tiles[tile_num - 1].click()
                    time.sleep(0.5)
            
            # Click the verify button
            verify_button = self.driver.find_element(By.CSS_SELECTOR, "#recaptcha-verify-button")
            verify_button.click()
            
            # Wait to see if we need to solve more challenges
            time.sleep(3)
            
            # Check if there are more challenges
            if self.driver.find_elements(By.CSS_SELECTOR, ".rc-imageselect-desc"):
                return self.solve_image_captcha()  # Recursively solve additional challenges
                
            # Switch back to default content
            self.driver.switch_to.default_content()
            return True
            
        except Exception as e:
            print(f"Error solving image CAPTCHA: {e}")
            # Switch back to default content
            self.driver.switch_to.default_content()
            return False
    
    def _parse_gemini_response(self, response_text):
        """Parse the Gemini API response to extract tile numbers."""
        try:
            # Look for sequences of numbers in the response
            import re
            numbers = re.findall(r'\d+', response_text)
            return [int(num) for num in numbers if 1 <= int(num) <= 9]
        except Exception as e:
            print(f"Error parsing AI response: {e}")
            return []
    
    def solve_captcha(self) -> bool:
        """Attempt to solve a detected CAPTCHA."""
        if not self.detect_captcha():
            return True  # No CAPTCHA to solve
            
        try:
            # First, try to find and click the checkbox
            try:
                # Switch to reCAPTCHA iframe if present
                captcha_frames = self.driver.find_elements(
                    By.CSS_SELECTOR, 
                    "iframe[src*='recaptcha'], iframe[title*='recaptcha'], iframe[title*='reCAPTCHA']"
                )
                
                if captcha_frames:
                    self.driver.switch_to.frame(captcha_frames[0])
                
                # Find and click the "I'm not a robot" checkbox
                checkbox = WebDriverWait(self.driver, 5).until(
                    EC.element_to_be_clickable((By.CSS_SELECTOR, ".recaptcha-checkbox-border"))
                )
                checkbox.click()
                time.sleep(2)
                
                # Switch back to main content
                self.driver.switch_to.default_content()
                
                # Check if we need to solve an image challenge
                image_frames = self.driver.find_elements(
                    By.CSS_SELECTOR, 
                    "iframe[src*='recaptcha/api2/bframe']"
                )
                
                if image_frames:
                    print("Image CAPTCHA challenge detected")
                    # Try our advanced image solving
                    if self.solve_image_captcha():
                        print("Successfully solved image CAPTCHA")
                    else:
                        print("Failed to solve image CAPTCHA, waiting for manual intervention")
                        # Wait longer for possible manual intervention
                        time.sleep(30)
                
                # Check if the CAPTCHA is still present
                time.sleep(3)
                if not self.detect_captcha():
                    print("CAPTCHA solved successfully")
                    return True
                    
            except (TimeoutException, NoSuchElementException) as e:
                print(f"Could not find standard reCAPTCHA elements: {e}")
            
            # Fallback: Try to use keyboard shortcuts to navigate and complete CAPTCHA
            print("Attempting keyboard navigation for CAPTCHA...")
            self.driver.find_element(By.TAG_NAME, 'body').send_keys('\t\t\t\t\t')  # Tab multiple times
            time.sleep(1)
            self.driver.find_element(By.TAG_NAME, 'body').send_keys('\r')  # Enter key
            time.sleep(3)
            
            return not self.detect_captcha()
            
        except Exception as e:
            print(f"Error solving CAPTCHA: {e}")
            return False


def get_captcha_solver(driver):
    """Factory function to create a CaptchaSolver instance."""
    return CaptchaSolver(driver)
